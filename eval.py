import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import CaptionDataset
from utils import collate_fn, create_captions_file, create_batched_graphs, console_log
import torch.nn.functional as F
from tqdm import tqdm
import dgl
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from models import cascade_sg_first_contextGAT_Decoder


def beam_evaluate(data_name, checkpoint_file, data_folder, beam_size, outdir, graph_feature_dim=512, dataset='TEST'):
    """
    Evaluation
    :param data_name: name of the data files
    :param checkpoint_file: which checkpoint file to use
    :param data_folder: folder where data is stored
    :param beam_size: beam size at which to generate captions for evaluation
    :param outdir: place where the outputs are stored, so the checkpoint file
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    global word_map
    device = torch.device("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    def load_dictionary():
        # Load word map (word2ix) using data folder provided
        word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}
        vocab_size = len(word_map)
        return word_map, rev_word_map, vocab_size
    
    word_map, rev_word_map, vocab_size = load_dictionary()
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
    # load state_dict
    torch.nn.Module.dump_patches = True
    stat_dict = torch.load(os.path.join(outdir, checkpoint_file), map_location=device)
    print('stat_dict scaler:', stat_dict['scaler'])
    scaler.load_state_dict(state_dict=stat_dict['scaler'])
    msg = decoder.load_state_dict(stat_dict['decoder'])
    print('messages in Decoder:', msg)
    print('missing_keys in Decoder:', msg.missing_keys)
    decoder = decoder.to(device)
    
    decoder.eval()
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, dataset),
        batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available())

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    with torch.no_grad():
        for caption_idx, (image_features, obj, rel, obj_mask, rel_mask, pair_ids, caps, caplens, orig_caps) in enumerate(
                loader):
            with torch.cuda.amp.autocast():
                if caption_idx % 5 != 0:
                    continue
        
                k = beam_size
        
        
                # Move to GPU device, if available
                image_features = image_features.to(device)  # (1, 36, 2048)
                obj = obj.to(device)
                rel = rel.to(device)
                obj_mask = obj_mask.to(device)
                rel_mask = rel_mask.to(device)
                # pair_ids = pair_ids.to(device)
                image_features_mean = image_features.mean(1)
                image_features_mean = image_features_mean.expand(k, 2048)
                graph_features_mean = torch.cat([obj, rel], dim=1).sum(dim=1) / \
                                      torch.cat([obj_mask, rel_mask], dim=1).sum(dim=1, keepdim=True)
                graph_features_mean = graph_features_mean.to(device)
                graph_features_mean = graph_features_mean.expand(k, graph_feature_dim)
        
                # initialize the graphs
                g = create_batched_graphs(obj, obj_mask, rel, rel_mask, pair_ids, beam_size=k)
                g = dgl.batch(g[:])
                # Tensor to store top k previous words at each step; now they're just <start>
                k_prev_words = torch.tensor([[word_map['<start>']]] * k, dtype=torch.long).to(device)  # (k, 1)
        
                # Tensor to store top k sequences; now they're just <start>
                seqs = k_prev_words  # (k, 1)
        
                # Tensor to store top k sequences' scores; now they're just 0
                top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        
                # Lists to store completed sequences and scores
                complete_seqs = list()
                complete_seqs_scores = list()
        
                # Start decoding
                step = 1
                h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
                h2, c2 = decoder.init_hidden_state(k)
        
                # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
                while True:
                    embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                    h1, c1 = decoder.top_down_attention(torch.cat([h2, image_features_mean, graph_features_mean, embeddings], dim=1),
                                                        (h1, c1))  # (batch_size_t, decoder_dim)
                    cgat_out, cgat_mask_out = decoder.context_gat(h1, g, batch_num_nodes=g.batch_num_nodes().tolist())
                    # make sure the size doesn't decrease
                    of = obj.repeat(cgat_out.size(0), 1, 1)
                    om = obj_mask.repeat(cgat_mask_out.size(0), 1)
                    cgat_obj = torch.zeros_like(of)  # size of number of objects
                    cgat_obj[:, :cgat_out.size(1)] = cgat_out  # fill with output of io attention
                    cgat_mask = torch.zeros_like(om)  # mask shaped like original objects
                    cgat_mask[:, :cgat_mask_out.size(1)] = cgat_mask_out  # copy over mask from io attention
                    cgat_obj[~cgat_mask & om] = of[~cgat_mask & om]  # fill the no in_degree nodes with the original state
                    # we pass the object mask. We used the cgat_mask only to determine which io's where filled and which not.
                    graph_weighted_enc = decoder.cascade1_attention(cgat_obj, h1, mask=om)
                    img_weighted_enc = decoder.cascade2_attention(image_features, torch.cat([h1, graph_weighted_enc], dim=1))
                    h2, c2 = decoder.language_model(torch.cat([graph_weighted_enc, img_weighted_enc, h1], dim=1), (h2, c2))
                    scores = decoder.fc(h2)  # (s, vocab_size)
                    scores = F.log_softmax(scores, dim=1)
                    # Add
                    scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        
                    # For the first step, all k points will have the same scores (since same k previous words, h, c)
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                    else:
                        # Unroll and find top scores, and their unrolled indices
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
        
                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = top_k_words // vocab_size  # (s)
                    next_word_inds = top_k_words % vocab_size  # (s)
        
                    # Add new words to sequences
                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        
                    # Which sequences are incomplete (didn't reach <end>)?
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)  # reduce beam length accordingly
        
                    # Proceed with incomplete sequences
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]
                    h1 = h1[prev_word_inds[incomplete_inds]]
                    c1 = c1[prev_word_inds[incomplete_inds]]
                    h2 = h2[prev_word_inds[incomplete_inds]]
                    c2 = c2[prev_word_inds[incomplete_inds]]
                    image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
                    graph_features_mean = graph_features_mean[prev_word_inds[incomplete_inds]]
                    gs = dgl.unbatch(g)
                    g = dgl.batch([gs[incomp_i] for incomp_i in incomplete_inds])
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                    # Break if things have been going on too long
                    if step > 50:
                        if len(complete_seqs) == 0:
                            # if we have to terminate, but none of the sequences are complete,
                            # recreate the complete inds without removing the incomplete ones: so everything.
                            complete_inds = list(set(range(len(next_word_inds))))
                            complete_seqs.extend(seqs[complete_inds].tolist())
                            complete_seqs_scores.extend(top_k_scores[complete_inds])
                        break
                    step += 1
    
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
    
            # References
            # img_caps = [' '.join(c) for c in orig_caps]
            img_caps = [c for c in orig_caps]
            references.append(img_caps)
    
            # Hypotheses
            hypothesis = (
            [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            # hypothesis = ' '.join(hypothesis)
            hypotheses.append(hypothesis)
            assert len(references) == len(hypotheses)

    # Calculate scores
    # metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    if not os.path.exists(os.path.join(outdir, 'hypotheses')):
        console.write_log('create dir of hypotheses', prefix='eval')
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
    if not os.path.exists(os.path.join(outdir, 'references')):
        console.write_log('create dir of references', prefix='eval')
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    hypotheses_file = os.path.join(outdir, 'hypotheses', '{}.{}.Hypotheses.json'.format(dataset,
                                                                                        data_name.split('_')[0]))
    references_file = os.path.join(outdir, 'references', '{}.{}.References.json'.format(dataset,
                                                                                        data_name.split('_')[0]))
    create_captions_file(range(len(hypotheses)), hypotheses, hypotheses_file)
    create_captions_file(range(len(references)), references, references_file)
    coco = COCO(references_file)
    # add the predicted results to the object
    coco_results = coco.loadRes(hypotheses_file)
    # create the evaluation object with both the ground-truth and the predictions
    coco_eval = COCOEvalCap(coco, coco_results)
    # change to use the image ids in the results object, not those from the ground-truth
    coco_eval.params['image_id'] = coco_results.getImgIds()
    # run the evaluation
    coco_eval.evaluate()
    # Results contains: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"
    results = coco_eval.eval
    return results

def eval_fast():
    hypotheses_file = 'outputs/ft50/hypotheses/TEST.coco.Hypotheses.json'
    references_file = 'outputs/ft50/references/TEST.coco.References.json'
    coco = COCO(references_file)
    # add the predicted results to the object
    coco_results = coco.loadRes(hypotheses_file)
    # create the evaluation object with both the ground-truth and the predictions
    coco_eval = COCOEvalCap(coco, coco_results)
    # change to use the image ids in the results object, not those from the ground-truth
    coco_eval.params['image_id'] = coco_results.getImgIds()
    # run the evaluation
    coco_eval.evaluate()
    # Results contains: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"
    results = coco_eval.eval
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/home/chunhui/dataset/mscoco/final_dataset', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--dataset', default='TEST', type=str, help='which split to use')
    parser.add_argument('--outdir', default='outputs/ft50/', type=str,
                        help='path to location where the outputs are saved, so the checkpoint')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', help="Checkpoint to use for beam search.")
    parser.add_argument('--beam_size', type=int, default=5,
            help="Beam size to use with beam search. If set to one we run greedy search.")
    parser.add_argument('--graph_feature_dim', type=int, default=512,
                        help="depends on which scene graph generator is used")
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3, 4, 5], default=0,
                        help="0: no augmentation, 1: node_drop, 2.sub_graph, 3.edge_drop, 4.attr_mask, 5.add_node.")
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
    args = parser.parse_args()
    cudnn.benchmark = True  # True only if inputs to model are fixed size, otherwise lot of computational overhead
    console = console_log(args.outdir)
    print(args)
    console.write_log(args.outdir, prefix='eval')
    metrics_dict = beam_evaluate(args.data_name, args.checkpoint_file, args.data_folder, args.beam_size, args.outdir, graph_feature_dim=args.graph_feature_dim, dataset=args.dataset)
    console.write_log(metrics_dict, prefix='eval')
