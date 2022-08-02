import argparse
import json
import os
import shutil
import time
import dgl
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import CaptionDataset, DataLoaderX, transform_img, transform_obj
from model_moco import MoCo
from utils import collate_fn, save_checkpoint, AverageMeter, adjust_learning_rate, accuracy, console_log, create_batched_graphs_augmented, accuracy_cl, AverageMeter_cl
# ggdG

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
    model = MoCo(K=args.K, args=args, vocab_size=len(word_map), mlp=args.mlp, word_map=word_map, teacher_force=False, freeze_embedding=args.freeze_embedding, console=console)
    # Move to GPU, if available
    model = model.to(device)
    model_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, model.parameters()))
    tracking = {'eval': [], 'test': None}
    start_epoch = 0
    best_epoch = -1
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation
    best_stopping_score = 0.  # stopping_score right now
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        args.stopping_metric = checkpoint['stopping_metric']
        best_stopping_score = checkpoint['metric_score']
        model_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        model.load_state_dict(checkpoint['decoder'])
        scaler.load_state_dict(checkpoint['scaler'])
        console.write_log('loading checkpoint successfully')
    

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)
    criterion_cl = nn.CrossEntropyLoss().to(device)
    
    # moco transforms
    moco_transforms = [transform_img, transform_obj]
    
    # Custom dataloaders
    train_loader = DataLoaderX(CaptionDataset(args.data_folder, args.data_name, 'TRAIN',
                                              two_crop=True, transform=moco_transforms),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == args.patience:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(model_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion_ce=criterion_ce,
              criterion_dis=criterion_dis,
              criterion_cl=criterion_cl,
              model_optimizer=model_optimizer,
              epoch=epoch, scaler=scaler)


        # Save checkpoint
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, model, model_optimizer,
                        args.stopping_metric, best_stopping_score, tracking, True, args.outdir, best_epoch, scaler)


def train(train_loader, model, criterion_ce, criterion_dis, criterion_cl, model_optimizer, epoch, scaler):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param model: MoCo model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param criterion_cl : contrast loss layer
    :param model_optimizer: optimizer to update model's weights
    :param epoch: epoch number
    """

    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    trn_loss_cl = AverageMeter()
    trn_loss_d = AverageMeter()
    top1_cl = AverageMeter()
    top5_cl = AverageMeter()
    start = time.time()

    # Batches
    for i, sample in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            data_time.update(time.time() - start)

            (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens) = sample
            # Move to GPU, if available
            imgs[0] = imgs[0].to(device, non_blocking=True)
            imgs[1] = imgs[1].to(device, non_blocking=True)
            # imgs = [imgs[0], imgs[1]]

            obj[0] = obj[0].to(device, non_blocking=True)
            obj[1] = obj[1].to(device, non_blocking=True)
            # obj = [imgs[0], imgs[1]]
            #
            # obj = obj.to(device)
            obj_mask = obj_mask.to(device, non_blocking=True)
            
            rel = rel.to(device, non_blocking=True)
            rel_mask = rel_mask.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            caplens = caplens.to(device, non_blocking=True)

            # Forward prop.
            logits, labels, scores, scores_d, caps_sorted, decode_lengths, sort_ind = model(imgs, obj, rel, obj_mask, rel_mask,
                                                                              pair_idx, caps, caplens)
            # Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]
            # print('scores:',scores.size(),'scores_d:',scores_d.size())
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0), scores_d.size(1), device="cuda")  # .to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:, :length - 1] = targets[:, :length - 1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # print('decode_lengths:', decode_lengths)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
            # Calculate loss
            loss_d = criterion_dis(scores_d, targets_d.long())
            # loss_g = criterion_ce(scores, targets)
            loss_cl = criterion_cl(logits, labels)
            loss = loss_d * 0.1 + loss_cl * 5
            acc1 = accuracy(logits, labels, 1)
            acc5 = accuracy(logits, labels, 5)
            top1_cl.update(acc1)
            top5_cl.update(acc5)
            trn_loss_cl.update(loss_cl.item())
            trn_loss_d.update(loss_d.item())
            # loss = 10 * loss_cl

        # Back prop.
        model_optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Clip gradients when they are getting too large
        scaler.unscale_(model_optimizer)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.25)

        # Update weights
        scaler.step(model_optimizer)
        scaler.update()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        with open(args.outdir + '/trn_loss_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.val)+'\n')
        with open(args.outdir + '/trn_loss_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.avg)+'\n')
        with open(args.outdir + '/trn_top1_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(top1_cl.val)+'\n')
        with open(args.outdir + '/trn_top1_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(top1_cl.avg)+'\n')
        with open(args.outdir + '/trn_top5_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5_cl.val)+'\n')
        with open(args.outdir + '/trn_top5_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5_cl.avg)+'\n')
        with open(args.outdir + '/trn_infoNCE.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_cl.item())+'\n')
        with open(args.outdir + '/trn_loss_d.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_d.item())+'\n')
            
        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            console.write_log('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'trn_loss_d {trn_loss_d.val:.4f} ({trn_loss_d.avg:.4f})\t'
                  'trn_loss_cl {trn_loss_cl.val:.4f} ({trn_loss_cl.avg:.4f})\t'
                  'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          trn_loss_d=trn_loss_d,
                                                                          trn_loss_cl=trn_loss_cl,
                                                                          top1=top1_cl, top5=top5_cl))

def validate(val_loader, model, criterion_ce, criterion_dis, criterion_cl, epoch=0):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param model: model.encoder_q
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param epoch: for which epoch is validated
    :return: BLEU-4 score
    """
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    val_loss_cl = AverageMeter()
    val_loss_d = AverageMeter()
    top1_cl = AverageMeter()
    top5_cl = AverageMeter()
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
                                          'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                                    batch_time=batch_time,
                                                                                                    loss=losses,
                                                                                                    top1=top1_cl, top5=top5_cl))
                    continue

                (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens, orig_caps) = sample
                # Move to GPU, if available
                imgs = imgs.to(device, non_blocking=True)
                obj = obj.to(device, non_blocking=True)
                obj_mask = obj_mask.to(device, non_blocking=True)
                rel = rel.to(device, non_blocking=True)
                rel_mask = rel_mask.to(device, non_blocking=True)
                caps = caps.to(device, non_blocking=True)
                caplens = caplens.to(device, non_blocking=True)

                # Forward prop.
                logits, labels, scores, scores_d, caps_sorted, decode_lengths, sort_ind = model(imgs, obj, rel, obj_mask, rel_mask,
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
                loss_cl = criterion_cl(logits, labels)
                loss = (5 * loss_d) + loss_cl

                acc1 = accuracy(logits, labels, 1)
                acc5 = accuracy(logits, labels, 5)
                top1_cl.update(acc1)
                top5_cl.update(acc5)
                val_loss_d.update(loss_d.item())
                val_loss_cl.update(loss_cl.item())
                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                batch_time.update(time.time() - start)

                with open(args.outdir + 'val_loss_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.val)+'\n')
                with open(args.outdir + 'val_loss_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.avg)+'\n')
                with open(args.outdir + 'trn_top1_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top1_cl.val)+'\n')
                with open(args.outdir + 'trn_top1_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top1_cl.avg)+'\n')
                with open(args.outdir + 'trn_top5_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5_cl.val)+'\n')
                with open(args.outdir + 'trn_top5_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5_cl.avg)+'\n')
                with open(args.outdir + 'val_infoNCE.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_cl.item())+'\n')
                with open(args.outdir + 'val_loss_d.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_d.item())+'\n')
                    
                start = time.time()

                if i % args.print_freq_val == 0:
                    console.write_log('Validation: [{0}/{1}]\t'
                                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                      'val_loss_d {val_loss_d.val:.4f} ({val_loss_d.avg:.4f})\t'
                                      'val_loss_cl {val_loss_cl.val:.4f} ({val_loss_cl.avg:.4f})\t'
                                      'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                                batch_time=batch_time,
                                                                                                loss=losses,
                                                                                                val_loss_d=val_loss_d, val_loss_cl=val_loss_cl,
                                                                                                top1=top1_cl, top5=top5_cl))

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

if __name__ == '__main__':
    print('start')
    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "loss", "top5"]
    parser = argparse.ArgumentParser('Image Captioning')
    # Add config file arguments
    parser.add_argument('--data_folder', default='/home/chunhui/dataset/mscoco/final_dataset', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--exp_name', default='moco', type=str, help='name under outputs/')
    parser.add_argument('--model_dir', default='outputs/moco/batch_size-512_epochs-52_dropout-0.5_patience-20_stop-metric-Bleu_4_aug-4_edgeprob-0.2_nodeprob-0.2_attrprob-0.2/emb-1024_att-1024_dec-1024/cgat_useobj-True_userel-True_ksteps-1_updaterel-True/seed-1/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', type=str,
                        help='base name shared by data files')
    parser.add_argument('--decoder_type', type=int, choices=[0, 1], default=1,
                        help="0: img_first, 1: sg_first.")
    parser.add_argument('--print_freq', default=128, type=int, help='print training stats every __ batches')
    parser.add_argument('--print_freq_val', default=1024, type=int, help='print validation stats every __ batches')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, None if none')
    parser.add_argument('--outdir', default='outputs/', type=str,
                        help='path to location where to save outputs. Empty for current working dir')
    parser.add_argument('--resume', default=False, type=bool, help='whether to reuse checkpoint file')
    parser.add_argument('--use_embedding', default=True, type=bool, help='whether to load embedding.pth.tar embedding layer')
    parser.add_argument('--freeze_embedding', default=True, type=bool, help='whether to freeze embedding layer')
    parser.add_argument('--embedding_bn', default=False, type=bool, help='whether to freeze embedding layer')
    parser.add_argument('--mlp', default=True, type=bool, help='whether to use mlp in moco')
    parser.add_argument('--workers', default=4, type=int,
                        help='for data-loading; right now, only 1 works with h5py ' 
                             '(OUTDATED, h5py can have multiple reads, right)')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3, 4, 5], default=3,
                        help="0: no augmentation, 1: node_drop, 2.sub_graph, 3.edge_drop, 4.attr_mask, 5.add_node.")
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--K', default=65536, type=int, help='number of negative sample, [65536, 131072]')
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
        console.write_log('SAVE_DIR will be deleted ...')
        shutil.rmtree(args.outdir)
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