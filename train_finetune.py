import argparse
import shutil
import time
import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import cascade_sg_first_contextGAT_Decoder
from datasets import CaptionDataset, DataLoaderX
from utils import collate_fn, save_checkpoint, AverageMeter, adjust_learning_rate, accuracy, create_captions_file, console_log
# from pycocotools.coco import COCO
# from pycocoevalcapalcap.eval import COCOEvalCap
# from eval import beam_evaluate
import dgl
from utils import create_batched_graphs, create_batched_graphs_augmented
import numpy as np
word_map = word_map_inv = None
scene_graph = False


def main():
    """
    Training and validation.
    """

    global word_map, word_map_inv, scene_graph

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        # create inverse word map
    word_map_inv = {v: k for k, v in word_map.items()}

    # Initialize / load checkpoint
    decoder = cascade_sg_first_contextGAT_Decoder(attention_dim=args.attention_dim,
                      embed_dim=args.emb_dim,
                      decoder_dim=args.decoder_dim,
                      graph_features_dim=args.graph_features_dim,
                      vocab_size=len(word_map),
                      dropout=args.dropout,
                      cgat_obj_info=args.cgat_obj_info,
                      cgat_rel_info=args.cgat_rel_info,
                      cgat_k_steps=args.cgat_k_steps,
                      cgat_update_rel=args.cgat_update_rel,
                      augmentation=args.augmentation,
                      edge_drop_prob=args.edge_drop_prob,
                      node_drop_prob=args.node_drop_prob,
                      attr_drop_prob=args.attr_drop_prob,
                      teacher_force=True,
                      embedding_bn=args.embedding_bn
                      )
    start_epoch = 0
    best_epoch = -1
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation
    best_stopping_score = 0.  # stopping_score right now
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # Move to GPU, if available
    decoder = decoder.to(device)
    decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
    if args.moco_ckpt is not None:
        console.write_log('loading moco_ckpt... ', args.moco_ckpt)
        checkpoint = torch.load(args.moco_ckpt)
        state_dict = checkpoint['decoder']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = decoder.load_state_dict(state_dict)
        console.write_log('messages in Decoder:', msg)
        console.write_log('missing_keys in Decoder:', msg.missing_keys)
        console.write_log('loading checkpoint successfully')
    elif args.resume:
        console.write_log('loading pretrained decoder_ckpt...', args.model_dir)
        checkpoint = torch.load(args.model_dir)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        tracking = checkpoint['tracking']
        best_epoch = checkpoint['best_epoch']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(state_dict=checkpoint['scaler'])
    else:
        raise ValueError('Invalid pretrained model of MoCo or Decoder. Plz check your ckpt path.')
    console.write_log('load successfully')


    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)

    # Custom dataloaders
    split_indices = np.load(''.join(['indices_dir/split_', str(args.split_rate), '_indices.npy'])).tolist()
    console.write_log('split_indices is: ', split_indices)
    trn_sampler = torch.utils.data.SubsetRandomSampler(split_indices)
    

    train_loader = DataLoaderX(CaptionDataset(args.data_folder, args.data_name, 'TRAIN'),
                               batch_size=args.batch_size, #shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True, sampler=trn_sampler)
    val_loader = DataLoaderX(CaptionDataset(args.data_folder, args.data_name, 'VAL',
                                            scene_graph=scene_graph),
                             collate_fn=collate_fn,
                             # use our specially designed collate function with valid/test only
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True, persistent_workers=True)
    best_valtop5acc = -999.9
    valtop5acc = -999.9
    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == args.patience:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              criterion_dis=criterion_dis,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch, scaler=scaler)
        # One epoch's validation
        if epoch >= 30:
            if epoch < 40 and epoch % 2 == 1:
                valtop5acc = validate(val_loader=val_loader,
                                      decoder=decoder,
                                      criterion_ce=criterion_ce,
                                      criterion_dis=criterion_dis,
                                      epoch=epoch)
            elif epoch < 40:
                console.write_log('skip val epoch ', epoch)
            else:
                valtop5acc = validate(val_loader=val_loader,
                                      decoder=decoder,
                                      criterion_ce=criterion_ce,
                                      criterion_dis=criterion_dis,
                                      epoch=epoch)
        else:
            console.write_log('skip val epoch ', epoch)

        is_best = valtop5acc > best_valtop5acc
        best_valtop5acc = max(valtop5acc, best_valtop5acc)
        if is_best:
            console.write_log("\n!!Best epoch: %d\n" % (epoch,))
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, args.stopping_metric, best_stopping_score, None, is_best, args.outdir, best_epoch, scaler)


def train(train_loader, decoder, criterion_ce, criterion_dis, decoder_optimizer, epoch, scaler):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, sample in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            data_time.update(time.time() - start)

            (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens) = sample
            # Move to GPU, if available
            imgs = imgs.to(device)
            obj = obj.to(device)
            obj_mask = obj_mask.to(device)
            rel = rel.to(device)
            rel_mask = rel_mask.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, obj, rel, obj_mask, rel_mask,
                                                                              pair_idx, caps, caplens)
            # Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0), scores_d.size(1), device="cuda")  # .to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:, :length - 1] = targets[:, :length - 1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
            # Calculate loss
            loss_d = criterion_dis(scores_d, targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss = loss_g + (10 * loss_d)

        # Back prop.
        decoder_optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Clip gradients when they are getting too large
        scaler.unscale_(decoder_optimizer)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        scaler.step(decoder_optimizer)
        scaler.update()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            console.write_log('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        with open(args.outdir + '/trn_loss_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.val)+'\n')
        with open(args.outdir + '/trn_loss_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.avg)+'\n')
        with open(args.outdir + '/trn_top5accs_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5accs.val)+'\n')
        with open(args.outdir + '/trn_top5accs_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5accs.avg)+'\n')
        with open(args.outdir + '/trn_loss_d.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_d.item()) + '\n')
        with open(args.outdir + '/trn_loss_g.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_g.item())+'\n')
            
def validate(val_loader, decoder, criterion_ce, criterion_dis, epoch=0):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param epoch: for which epoch is validated
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        # for i, (imgs, caps, caplens,allcaps) in enumerate(val_loader):
        for i, sample in enumerate(val_loader):
            with torch.cuda.amp.autocast():
                if i % 5 != 0:
                    # only decode every 5th caption, starting from idx 0.
                    # this is because the iterator iterates over all captions in the dataset, not all images.
                    if i % args.print_freq_val == 0:
                        console.write_log('Validation: [{0}/{1}]\t'
                                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                                    batch_time=batch_time,
                                                                                                    loss=losses,
                                                                                                    top5=top5accs))
                    continue

                (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens, orig_caps) = sample
                # Move to GPU, if available
                imgs = imgs.to(device)
                obj = obj.to(device)
                obj_mask = obj_mask.to(device)
                rel = rel.to(device)
                rel_mask = rel_mask.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # Forward prop.
                scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, obj, rel, obj_mask, rel_mask,
                                                                                  pair_idx, caps, caplens)

                # Max-pooling across predicted words across time steps for discriminative supervision
                scores_d = scores_d.max(1)[0]

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]
                targets_d = torch.zeros(scores_d.size(0), scores_d.size(1), device="cuda")  # .to(device)
                targets_d.fill_(-1)

                for length in decode_lengths:
                    targets_d[:, :length - 1] = targets[:, :length - 1]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data

                # Calculate loss
                loss_d = criterion_dis(scores_d, targets_d.long())
                loss_g = criterion_ce(scores, targets)
                loss = loss_g + (10 * loss_d)

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % args.print_freq_val == 0:
                    console.write_log('Validation: [{0}/{1}]\t'
                                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                                batch_time=batch_time,
                                                                                                loss=losses,
                                                                                                top5=top5accs))
                with open(args.outdir + '/val_loss_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.val) + '\n')
                with open(args.outdir + '/val_loss_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.avg) + '\n')
                with open(args.outdir + '/val_top5accs_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5accs.val) + '\n')
                with open(args.outdir + '/val_top5accs_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5accs.avg) + '\n')
                with open(args.outdir + '/val_loss_d.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_d.item()) + '\n')
                with open(args.outdir + '/val_loss_g.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_g.item()) + '\n')

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

                # References
                assert (len(sort_ind) == 1), "Cannot have batch_size>1 for validation."
                # a reference is a list of lists:
                # [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['a', 'cat', 'on', 'the', 'mat']]
                references.append(orig_caps)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                preds_idxs_no_pads = list()
                for j, p in enumerate(preds):
                    preds_idxs_no_pads.append(preds[j][:decode_lengths[j]])  # remove pads
                    preds_idxs_no_pads = list(map(lambda c: [w for w in c if w not in {word_map['<start>'],
                                                                                       word_map['<pad>']}],
                                                  preds_idxs_no_pads))
                temp_preds = list()
                # remove <start> and pads and convert idxs to string
                for hyp in preds_idxs_no_pads:
                    temp_preds.append([])
                    for w in hyp:
                        assert (not w == word_map['<pad>']), "Should have removed all pads."
                        if not w == word_map['<start>']:
                            temp_preds[-1].append(word_map_inv[w])
                preds = temp_preds
                hypotheses.extend(preds)
                assert len(references) == len(hypotheses)

    return top5accs.avg
    # Calculate BLEU-4 scores
    # compute the metrics
    # hypotheses_file = os.path.join(args.outdir, 'hypotheses', 'Epoch{:0>3d}.Hypotheses.json'.format(epoch))
    # references_file = os.path.join(args.outdir, 'references', 'Epoch{:0>3d}.References.json'.format(epoch))
    # create_captions_file(range(len(hypotheses)), hypotheses, hypotheses_file)
    # create_captions_file(range(len(references)), references, references_file)
    # coco = COCO(references_file)
    # # add the predicted results to the object
    # coco_results = coco.loadRes(hypotheses_file)
    # # create the evaluation object with both the ground-truth and the predictions
    # coco_eval = COCOEvalCap(coco, coco_results)
    # # change to use the image ids in the results object, not those from the ground-truth
    # coco_eval.params['image_id'] = coco_results.getImgIds()
    # # run the evaluation
    # coco_eval.evaluate(verbose=False, metrics=['bleu', 'meteor', 'rouge', 'cider'])
    # # Results contains: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"
    # results = coco_eval.eval
    # results['loss'] = losses.avg
    # results['top5'] = top5accs.avg
    #
    # for k, v in results.items():
    #     console.write_log(k+':\t'+str(v))
    # # print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}, CIDEr - {cider}\n'
    # #       .format(loss=losses, top5=top5accs, bleu=round(results['Bleu_4'], 4), cider=round(results['CIDEr'], 1)))
    # return results


if __name__ == '__main__':
    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "loss", "top5"]
    parser = argparse.ArgumentParser('Image Captioning')
    # Add config file arguments
    parser.add_argument('--data_folder', default='/home/chunhui/dataset/mscoco/final_dataset', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--exp_name', default='retrain_cl', type=str, help='name under outputs/')
    parser.add_argument('--split_rate', default='01', type=str, help='split_rate for finetune')
    parser.add_argument('--model_dir', default=None, type=str, help='base name shared by data files')
    parser.add_argument('--decoder_type', type=int, choices=[0, 1], default=1,
                        help="0: img_first, 1: sg_first.")
    parser.add_argument('--print_freq', default=128, type=int, help='print training stats every __ batches')
    parser.add_argument('--print_freq_val', default=1024, type=int, help='print validation stats every __ batches')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, None if none')
    parser.add_argument('--outdir', default='outputs/', type=str,
                        help='path to location where to save outputs. Empty for current working dir')
    parser.add_argument('--resume', default=False, type=bool, help='whether to reuse checkpoint file')
    parser.add_argument('--moco_ckpt', default=None, type=str, help='path to moco pretrain checkpoint, None if none')
    parser.add_argument('--workers', default=4, type=int,
                        help='for data-loading; right now, only 1 works with h5py ' 
                             '(OUTDATED, h5py can have multiple reads, right)')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3, 4, 5], default=4,
                        help="0: no augmentation, 1: node_drop, 2.sub_graph, 3.edge_drop, 4.attr_mask, 5.add_node.")
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--seed', default=1, type=int, help='The random seed that will be used.')
    parser.add_argument('--edge_drop_prob', default=0.2, type=float, help='edge_drop_prob')
    parser.add_argument('--node_drop_prob', default=0.2, type=float, help='node_drop_prob')
    parser.add_argument('--attr_drop_prob', default=0.2, type=float, help='attr_drop_prob')
    parser.add_argument('--emb_dim', default=1024, type=int, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', default=1024, type=int, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', default=1024, type=int, help='dimension of decoder lstm layers')
    parser.add_argument('--graph_features_dim', default=512, type=int, help='dimension of graph features')
    parser.add_argument('--cgat_obj_info', default=True, type=bool, help='whether to use object info in CGAT')
    parser.add_argument('--cgat_rel_info', default=True, type=bool, help='whether to use relation info in CGAT')
    parser.add_argument('--cgat_k_steps', default=1, type=bool, help='how many CGAT steps to do')
    parser.add_argument('--cgat_update_rel', default=True, type=bool, help='whether to update relation states '
                                                                           'for k CGAT steps')
    parser.add_argument('--dropout', default=0.5, type=float, help='dimension of decoder RNN')
    parser.add_argument('--embedding_bn', default=False, type=bool, help='whether to freeze embedding layer')
    parser.add_argument('--epochs', default=52, type=int,
                        help='number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--patience', default=20, type=int,
                        help='stop training when metric doesnt improve for this many epochs')
    parser.add_argument('--stopping_metric', default='Bleu_4', type=str, choices=metrics,
                        help='which metric to use for early stopping')
    parser.add_argument('--test_at_end', default=True, type=bool, help='If there should be tested on the test split')
    parser.add_argument('--beam_size', default=5, type=int, help='If test at end, beam size to use for testing.')

    # Parse the arguments
    args = parser.parse_args()

    device = torch.device("cuda")  # sets device for model and PyTorch tensors
    # setup initial stuff for reproducability
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size otherwise lot of computational overhead
    torch.manual_seed(args.seed)

    args.outdir = os.path.join(args.outdir,
                               args.exp_name,
                               'batch_size-{bs}_epochs-{ep}_dropout-{drop}_patience-{pat}_stop-metric-{met}_aug-{augmentation}_edgeprob-{edge_drop_prob}_nodeprob-{node_drop_prob}_attrprob-{attr_drop_prob}'.format(
                                   bs=args.batch_size, ep=args.epochs, drop=args.dropout,
                                   pat=args.patience, met=args.stopping_metric,augmentation=args.augmentation,
                                                                      edge_drop_prob=args.edge_drop_prob,
                                                                      node_drop_prob=args.node_drop_prob,
                                                                      attr_drop_prob=args.attr_drop_prob),
                               'emb-{emb}_att-{att}_dec-{dec}'.format(emb=args.emb_dim, att=args.attention_dim,
                                                                      dec=args.decoder_dim),
                               'cgat_useobj-{o}_userel-{r}_ksteps-{k}_updaterel-{u}'.format(
                                   o=args.cgat_obj_info, r=args.cgat_rel_info, k=args.cgat_k_steps,
                                   u=args.cgat_update_rel),
                               'seed-{}'.format(args.seed))

    console = console_log()
    if os.path.exists(args.outdir) and args.checkpoint is None:
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    elif os.path.exists(args.outdir) and args.checkpoint is not None:
        console.write_log('continueing from checkpoint {} in {}...'.format(args.checkpoint, args.outdir))
    elif not os.path.exists(args.outdir) and args.checkpoint is not None:
        console.write_log('set a checkpoint to continue from, but the save directory from --outdir {} does not exist. '
              'setting --checkpoint to None'.format(args.outdir))
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    console = console_log(args.outdir)
    print(args)
    console.write_log(args.outdir)
    main()
