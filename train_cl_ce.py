print('before import...')
import argparse
import json
import os
import shutil
import time
from tqdm import tqdm
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
# from eval import beam_evaluate
import dgl
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from datasets import CaptionDataset, DataLoaderX, transform_img, transform_obj
from model_moco import MoCo
from utils import collate_fn, save_checkpoint, AverageMeter, adjust_learning_rate, accuracy, console_log, create_batched_graphs_augmented, accuracy_cl, AverageMeter_cl
# from torch.utils.tensorboard import SummaryWriter
print('import...')

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
    model = MoCo(args=args, vocab_size=len(word_map), mlp=True, word_map=word_map, teacher_force=False, freeze_embedding = args.freeze_embedding, console=console)
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
    val_loader = DataLoaderX(CaptionDataset(args.data_folder, args.data_name, 'VAL',
                                            scene_graph=scene_graph),
                             collate_fn=collate_fn,
                             # use our specially designed collate function with valid/test only
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=False, persistent_workers=True)

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
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, model, model_optimizer, args.stopping_metric, best_stopping_score, tracking, True, args.outdir, best_epoch, scaler)
        # One epoch's validation
        if epoch % 3 == 1:
            validate(val_loader=val_loader,
                     model=model.encoder_q,
                     criterion_ce=criterion_ce,
                     criterion_dis=criterion_dis,
                     epoch=epoch)
        elif epoch >= 30:
            if epoch < 40 and epoch % 2 == 1:
                validate(val_loader=val_loader,
                         model=model.encoder_q,
                         criterion_ce=criterion_ce,
                         criterion_dis=criterion_dis,
                         epoch=epoch)
            elif epoch < 40:
                print('skip val epoch ', epoch)
            else:
                validate(val_loader=val_loader,
                         model=model.encoder_q,
                         criterion_ce=criterion_ce,
                         criterion_dis=criterion_dis,
                         epoch=epoch)
        else:
            print('skip val epoch ', epoch)
        # tracking['eval'].append(recent_results)
        # recent_stopping_score = recent_results[args.stopping_metric]

        # Check if there was an improvement
        # is_best = recent_stopping_score > best_stopping_score
        # best_stopping_score = max(recent_stopping_score, best_stopping_score)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     console.write_log("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0
        #     best_epoch = epoch

        # Save checkpoint

    # if needed, run an beamsearch evaluation on the test set
    # if args.test_at_end:
    #     checkpoint_file = 'BEST_' + str(best_epoch) + '_' + 'checkpoint_' + args.data_name + '.pth.tar'
    #     results = beam_evaluate(args.data_name, checkpoint_file, args.data_folder, args.beam_size, args.outdir,
    #                             args.graph_features_dim)
    #     tracking['test'] = results
    # with open(os.path.join(args.outdir, 'TRACKING.' + args.data_name + '.pkl'), 'wb') as f:
    #     pickle.dump(tracking, f)

def load_model(path):
    # Load model using checkpoint file provided
    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(path, map_location=torch.device("cuda"))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    global word_map, word_map_inv, scene_graph
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        # create inverse word map
    word_map_inv = {v: k for k, v in word_map.items()}
    val_loader = DataLoaderX(CaptionDataset(args.data_folder, args.data_name, 'VAL',
                                            scene_graph=scene_graph),
                             collate_fn=collate_fn,
                             # use our specially designed collate function with valid/test only
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True, persistent_workers=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.load_state_dict(state_dict=checkpoint['scaler'])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)
    eval_validate(val_loader=val_loader,
             decoder=decoder,
             criterion_ce=criterion_ce,
             criterion_dis=criterion_dis,)

def eval_validate(val_loader, decoder, criterion_ce, criterion_dis):
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
                imgs = imgs.to(device, non_blocking=True)
                obj = obj.to(device, non_blocking=True)
                obj_mask = obj_mask.to(device, non_blocking=True)
                rel = rel.to(device, non_blocking=True)
                rel_mask = rel_mask.to(device, non_blocking=True)
                caps = caps.to(device, non_blocking=True)
                caplens = caplens.to(device, non_blocking=True)

                # Forward prop.

                def decoder_forward(decoder, image_features, object_features, relation_features, object_mask, relation_mask,
                            pair_ids,
                            encoded_captions, caption_lengths):
                    """
                    Forward propagation.
                    :param image_features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
                    :param graph_features: encoded images as graphs, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
                    :param graph_mask: mask for the graph_features, shows were non empty features are
                    :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
                    :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
                    :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
                    """
                    print('decoder_forward')
                    batch_size = image_features.size(0)
                    vocab_size = decoder.vocab_size

                    # Flatten image
                    image_features_mean = image_features.mean(1).to(device, non_blocking=True)  # (batch_size, num_pixels, encoder_dim)
                    graph_features_mean = torch.cat([object_features, relation_features], dim=1).sum(dim=1) / \
                                          torch.cat([object_mask, relation_mask], dim=1).sum(dim=1, keepdim=True)
                    graph_features_mean = graph_features_mean.to(device, non_blocking=True)

                    # Sort input data by decreasing lengths; why? apparent below
                    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
                    image_features = image_features[sort_ind]
                    object_features = object_features[sort_ind]
                    object_mask = object_mask[sort_ind]
                    relation_features = relation_features[sort_ind]
                    relation_mask = relation_mask[sort_ind]
                    pair_ids = pair_ids[sort_ind]
                    image_features_mean = image_features_mean[sort_ind]
                    graph_features_mean = graph_features_mean[sort_ind]
                    encoded_captions = encoded_captions[sort_ind]

                    # initialize the graphs
                    g, object_features, object_mask = create_batched_graphs_augmented(object_features, object_mask,
                                                                                      relation_features,
                                                                                      relation_mask,
                                                                                      pair_ids,
                                                                                      augmentation=0,
                                                                                      edge_drop_prob=decoder.edge_drop_prob,
                                                                                      node_drop_prob=decoder.node_drop_prob,
                                                                                      attr_drop_prob=decoder.attr_drop_prob)
                    # Embedding
                    embeddings = decoder.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

                    # Initialize LSTM state
                    h1, c1 = decoder.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
                    h2, c2 = decoder.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

                    # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
                    # So, decoding lengths are actual lengths - 1
                    decode_lengths = (caption_lengths - 1).tolist()

                    # Create tensors to hold word predicion scores
                    predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size, device="cuda")  # .to(device)
                    predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size,
                                               device="cuda")  # .to(device)

                    # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
                    # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
                    # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
                    # are then passed to the language model
                    for t in range(max(decode_lengths)):
                        batch_size_t = sum([l > t for l in decode_lengths])
                        sub_g = dgl.batch(g[:batch_size_t])
                        h1, c1 = decoder.top_down_attention(torch.cat([h2[:batch_size_t],
                                                                    image_features_mean[:batch_size_t],
                                                                    graph_features_mean[:batch_size_t],
                                                                    embeddings[:batch_size_t, t, :]], dim=1),
                                                         (h1[:batch_size_t], c1[:batch_size_t]))
                        cgat_out, cgat_mask_out = decoder.context_gat(h1[:batch_size_t], sub_g,
                                                                   batch_num_nodes=sub_g.batch_num_nodes().tolist())
                        # make sure the size doesn't decrease
                        of = object_features[:batch_size_t]
                        om = object_mask[:batch_size_t]
                        cgat_obj = torch.zeros_like(of)  # size of number of objects
                        cgat_obj[:, :cgat_out.size(1)] = cgat_out  # fill with output of cgat
                        cgat_mask = torch.zeros_like(om)  # mask shaped like original objects
                        cgat_mask[:, :cgat_mask_out.size(1)] = cgat_mask_out  # copy over mask from cgat
                        cgat_obj[~cgat_mask & om] = of[
                            ~cgat_mask & om]  # fill the no in_degree nodes with the original state
                        # we pass the object mask. We used the cgat_mask only to determine which io's where filled and which not.
                        graph_weighted_enc = decoder.cascade1_attention(cgat_obj[:batch_size_t], h1[:batch_size_t],
                                                                     mask=om)
                        img_weighted_enc = decoder.cascade2_attention(image_features[:batch_size_t],
                                                                   torch.cat([h1[:batch_size_t],
                                                                              graph_weighted_enc[:batch_size_t]],
                                                                             dim=1))
                        preds1 = decoder.fc1(decoder.dropout(h1))

                        h2, c2 = decoder.language_model(
                            torch.cat(
                                [graph_weighted_enc[:batch_size_t], img_weighted_enc[:batch_size_t], h1[:batch_size_t]],
                                dim=1),
                            (h2[:batch_size_t], c2[:batch_size_t]))
                        preds = decoder.fc(decoder.dropout(h2))  # (batch_size_t, vocab_size)
                        predictions[:batch_size_t, t, :] = preds
                        predictions1[:batch_size_t, t, :] = preds1

                    return predictions, predictions1, encoded_captions, decode_lengths, sort_ind

                # forward
                scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder_forward(decoder, imgs, obj, rel, obj_mask, rel_mask,
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
    trn_loss_ce = AverageMeter()
    top1_cl = AverageMeter()
    top5_cl = AverageMeter()
    top1_ce = AverageMeter()
    top5_ce = AverageMeter()
    start = time.time()

    for i, sample in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            data_time.update(time.time() - start)

            (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens) = sample
            # Move to GPU, if available
            imgs[0] = imgs[0].to(device, non_blocking=True)
            imgs[1] = imgs[1].to(device, non_blocking=True)

            obj[0] = obj[0].to(device, non_blocking=True)
            obj[1] = obj[1].to(device, non_blocking=True)
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
            # print('decode_lengths:', decode_lengths)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
            # Calculate loss
            loss_d = criterion_dis(scores_d, targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss_cl = criterion_cl(logits, labels)
            loss = loss_g + (5 * loss_d) + (5 * loss_cl)
            acc1_cl = accuracy(logits, labels, 1)
            acc5_cl = accuracy(logits, labels, 5)
            acc1_ce = accuracy(scores, targets, 1)
            acc5_ce = accuracy(scores, targets, 5)
            top1_cl.update(acc1_cl)
            top5_cl.update(acc5_cl)
            top1_ce.update(acc1_ce)
            top5_ce.update(acc5_ce)
            trn_loss_cl.update(loss_cl.item())
            trn_loss_d.update(loss_d.item())
            trn_loss_ce.update(loss_g.item())

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

        with open(args.outdir + 'trn_loss_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.val)+'\n')
        with open(args.outdir + 'trn_loss_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(losses.avg)+'\n')
        with open(args.outdir + 'trn_top1_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(top1_cl.val)+'\n')
        with open(args.outdir + 'trn_top1_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(top1_cl.avg)+'\n')
        with open(args.outdir + 'trn_top5_val.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5_cl.val)+'\n')
        with open(args.outdir + 'trn_top5_avg.txt', 'a', encoding='utf-8') as f:
            f.write(str(top5_cl.avg)+'\n')
        with open(args.outdir + 'trn_infoNCE.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_cl.item())+'\n')
        with open(args.outdir + 'trn_loss_d.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_d.item()) + '\n')
        with open(args.outdir + 'trn_loss_ce.txt', 'a', encoding='utf-8') as f:
            f.write(str(loss_g.item()) +'\n')
            
        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            console.write_log('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'trn_loss_d {trn_loss_d.val:.4f} ({trn_loss_d.avg:.4f})\t'
                  'trn_loss_cl {trn_loss_cl.val:.4f} ({trn_loss_cl.avg:.4f})\t'
                  'trn_loss_ce {trn_loss_ce.val:.4f} ({trn_loss_ce.avg:.4f})\t'
                  'Top-1ce Accuracy {top1ce.val:.3f} ({top1ce.avg:.3f})\t'
                  'Top-1ce Accuracy {top5ce.val:.3f} ({top5ce.avg:.3f})\t'
                  'Top-1cl Accuracy {top1cl.val:.3f} ({top1cl.avg:.3f})\t'
                  'Top-5cl Accuracy {top5cl.val:.3f} ({top5cl.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          trn_loss_d=trn_loss_d,
                                                                          trn_loss_cl=trn_loss_cl,
                                                                          trn_loss_ce=trn_loss_ce,
                                                                          top1ce=top1_ce, top5ce=top5_ce,
                                                                          top1cl=top1_cl, top5cl=top5_cl))

def validate(val_loader, model, criterion_ce, criterion_dis, epoch=0):
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
    val_loss_d = AverageMeter()
    val_loss_ce = AverageMeter()
    top1_ce = AverageMeter()
    top5_ce = AverageMeter()
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
                                          'Top-1 Accuracy {top1ce.val:.3f} ({top1ce.avg:.3f})\t'
                                          'Top-1 Accuracy {top5ce.val:.3f} ({top5ce.avg:.3f})\t'
                                          .format(i, len(val_loader),
                                                                                                    batch_time=batch_time,
                                                                                                    loss=losses,
                                                                                                    top1ce=top1_ce, top5ce=top5_ce))
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
                scores, scores_d, caps_sorted, decode_lengths, sort_ind = model(imgs, obj, rel, obj_mask, rel_mask,
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

                acc1_ce = accuracy(scores, targets, 1)
                acc5_ce = accuracy(scores, targets, 5)
                top1_ce.update(acc1_ce)
                top5_ce.update(acc5_ce)
                val_loss_d.update(loss_d.item())
                val_loss_ce.update(loss_g.item())
                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                batch_time.update(time.time() - start)
 
                with open(args.outdir + 'val_loss_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.val)+'\n')
                with open(args.outdir + 'val_loss_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(losses.avg)+'\n')
                with open(args.outdir + 'val_loss_d.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_d.item())+'\n')
                with open(args.outdir + 'val_loss_ce.txt', 'a', encoding='utf-8') as f:
                    f.write(str(loss_g.item())+'\n')
                with open(args.outdir + 'val_top1_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top1_ce.val) + '\n')
                with open(args.outdir + 'val_top1_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top1_ce.avg) + '\n')
                with open(args.outdir + 'val_top5_val.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5_ce.val) + '\n')
                with open(args.outdir + 'val_top5_avg.txt', 'a', encoding='utf-8') as f:
                    f.write(str(top5_ce.avg) + '\n')
                start = time.time()

                if i % args.print_freq_val == 0:
                    console.write_log('Validation: [{0}/{1}]\t'
                                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                      'val_loss_d {val_loss_d.val:.4f} ({val_loss_d.avg:.4f})\t'
                                      'val_loss_ce {val_loss_ce.val:.4f} ({val_loss_ce.avg:.4f})\t'
                                      'Top-1ce Accuracy {top1ce.val:.3f} ({top1ce.avg:.3f})\t'
                                      'Top-5ce Accuracy {top5ce.val:.3f} ({top5ce.avg:.3f})\t'.format(i, len(val_loader),
                                                                                                batch_time=batch_time,
                                                                                                loss=losses,
                                                                                                top1ce=top1_ce, top5ce=top5_ce,
                                                                                                val_loss_d=val_loss_d, val_loss_ce=val_loss_ce))

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
    parser.add_argument('--exp_name', default='clce', type=str, help='name under outputs/')
    parser.add_argument('--model_dir', default='outputs/cascade_img_first_contextGAT/batch_size-512_epochs-52_dropout-0.5_patience-20_stop-metric-Bleu_4_aug-4_edgeprob-0.2_nodeprob-0.2_attrprob-0.2/emb-1024_att-1024_dec-1024/cgat_useobj-True_userel-True_ksteps-1_updaterel-True/seed-1/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', type=str,
                        help='base name shared by data files')
    parser.add_argument('--decoder_type', type=int, choices=[0, 1], default=1,
                        help="0: img_first, 1: sg_first.")
    parser.add_argument('--print_freq', default=128, type=int, help='print training stats every __ batches')
    parser.add_argument('--print_freq_val', default=1024, type=int, help='print validation stats every __ batches')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, None if none')
    parser.add_argument('--outdir', default='outputs/', type=str,
                        help='path to location where to save outputs. Empty for current working dir')
    parser.add_argument('--resume', default=False, type=bool, help='whether to reuse checkpoint file')
    parser.add_argument('--freeze_embedding', default=True, type=bool, help='whether to freeze embedding layer')
    parser.add_argument('--embedding_bn', default=False, type=bool, help='whether to freeze embedding layer')
    parser.add_argument('--workers', default=4, type=int,
                        help='for data-loading; right now, only 1 works with h5py ' 
                             '(OUTDATED, h5py can have multiple reads, right)')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3, 4, 5], default=3,
                        help="0: no augmentation, 1: node_drop, 2.sub_graph, 3.edge_drop, 4.attr_mask, 5.add_node.")
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
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
    parser.add_argument('--dropout', default=0.1, type=float, help='dimension of decoder RNN')
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
    # cudnn.benchmark = True  # set to true only if inputs to model are fixed size otherwise lot of computational overhead
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
    console.write_log(args.outdir)
    main()
    # load_model(args.model_dir)