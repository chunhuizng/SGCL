import torch
import torch.nn as nn
from models import cascade_sg_first_contextGAT_Decoder


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=9490, K=65536, m=0.999, T=0.07, mlp=True, args=None, vocab_size=None, word_map=None, embedding_path='embedding.pth.tar', use_embedding=True, teacher_force=False, freeze_embedding=True, console=None):
        """
        dim: feature dimension (default: [batch_size, 10, 9490])
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.dim=dim
        self.K = K
        self.m = m
        self.T = T
        self.word_map = word_map

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = cascade_sg_first_contextGAT_Decoder(attention_dim=args.attention_dim,
                                                          embed_dim=args.emb_dim,
                                                          decoder_dim=args.decoder_dim,
                                                          graph_features_dim=args.graph_features_dim,
                                                          vocab_size=vocab_size,
                                                          dropout=args.dropout,
                                                          cgat_obj_info=args.cgat_obj_info,
                                                          cgat_rel_info=args.cgat_rel_info,
                                                          cgat_k_steps=args.cgat_k_steps,
                                                          cgat_update_rel=args.cgat_update_rel,
                                                          augmentation=args.augmentation,
                                                          edge_drop_prob=args.edge_drop_prob,
                                                          node_drop_prob=args.node_drop_prob,
                                                          attr_drop_prob=args.attr_drop_prob,
                                                          word_map=word_map,
                                                          teacher_force=teacher_force,
                                                          embedding_bn=args.embedding_bn
                                                          )
        self.encoder_k = cascade_sg_first_contextGAT_Decoder(attention_dim=args.attention_dim,
                                                          embed_dim=args.emb_dim,
                                                          decoder_dim=args.decoder_dim,
                                                          graph_features_dim=args.graph_features_dim,
                                                          vocab_size=vocab_size,
                                                          dropout=args.dropout,
                                                          cgat_obj_info=args.cgat_obj_info,
                                                          cgat_rel_info=args.cgat_rel_info,
                                                          cgat_k_steps=args.cgat_k_steps,
                                                          cgat_update_rel=args.cgat_update_rel,
                                                          augmentation=args.augmentation,
                                                          edge_drop_prob=args.edge_drop_prob,
                                                          node_drop_prob=args.node_drop_prob,
                                                          attr_drop_prob=args.attr_drop_prob,
                                                          word_map = word_map,
                                                          teacher_force=teacher_force,
                                                          embedding_bn=args.embedding_bn
                                                             )

        self.mlp = mlp
        if mlp:  # hack: brute-force replacement
            self.register_buffer("queue", torch.randn(self.encoder_q.projection_dim, K))
        else:
            self.register_buffer("queue", torch.randn(self.dim, K))
        
        if use_embedding:
            embedding_dict = torch.load(embedding_path)
            self.encoder_q.embedding.load_state_dict(embedding_dict)
            console.write_log('load embedding successfully')
            if args.freeze_embedding==True:
                for param_embedding in self.encoder_q.embedding.parameters():
                    param_embedding.requires_grad = False
                console.write_log("freeze embedding layer before training...")
            else:
                console.write_log("do not freeze embedding layer")
        else:
            console.write_log('do not load embedding.pth.tar')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.m = nn.AdaptiveAvgPool1d(1)
        # create the queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print('keys.T.size()', keys.T.size())
        # print('self.queue[..., ptr:ptr + batch_size]',self.queue[..., ptr:ptr + batch_size].size())
        self.queue[..., ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, image_features_k, object_features_k, relation_features, object_mask, relation_mask, pair_ids,
                                  encoded_captions, caption_lengths):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support 1 gpu model. ***
        """
        # gather from 1 gpu
        batch_size_all = image_features_k.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return image_features_k[idx_shuffle], object_features_k[idx_shuffle], relation_features[idx_shuffle], object_mask[idx_shuffle], relation_mask[idx_shuffle], pair_ids[idx_shuffle], \
               encoded_captions[idx_shuffle], caption_lengths[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
		Undo batch shuffle.
		*** Only support 1 gpu model. ***
		"""
        # gather from 1 gpu

        return x[idx_unshuffle]
    
    def forward(self,image_features, object_features, relation_features, object_mask, relation_mask, pair_ids,
                encoded_captions, caption_lengths):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        image_features_q = image_features[0]
        object_features_q = object_features[0]

        image_features_k = image_features[1]
        object_features_k = object_features[1]
        # compute query features
        scores, scores_d, caps_sorted, decode_lengths, sort_ind = self.encoder_q(image_features_q, object_features_q, relation_features, object_mask, relation_mask, pair_ids,
                encoded_captions, caption_lengths)  # queries: NxC
        
        q = torch.sum(scores, dim=1) / scores.size(1)
        if self.mlp:  # hack: brute-force replacement
            q = self.encoder_q.projection(q)
            q = self.encoder_q.projection1(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            image_features_k, object_features_k, relation_features, object_mask, relation_mask, pair_ids,\
            encoded_captions, caption_lengths, idx_unshuffle = self._batch_shuffle_single_gpu(image_features_k, object_features_k, relation_features, object_mask, relation_mask, pair_ids,
                encoded_captions, caption_lengths)

            _scores_k, _scores_d, _caps_sorted, _decode_lengths, _sort_ind = self.encoder_k(image_features_k, object_features_k, relation_features, object_mask, relation_mask, pair_ids,
                encoded_captions, caption_lengths)  # keys: NxC
            k = torch.sum(_scores_k, dim=1) / _scores_k.size(1)
            if self.mlp:  # hack: brute-force replacement
                k = self.encoder_k.projection(k)
                k = self.encoder_k.projection1(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        
        # print('caps_sorted:',caps_sorted.size(),'_caps_sorted:',_caps_sorted)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, scores, scores_d, caps_sorted, decode_lengths, sort_ind


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output