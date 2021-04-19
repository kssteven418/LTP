# Copyright 2021 Samsung Semiconductor Incorporated.

import torch


class CascadeTokenPruner:
    """
    implements the layer-by-layer operations for the cascade token pruning method described in:

    Wang et al.,
    SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning
    https://arxiv.org/abs/2012.09852
    """
    def __init__(self):
        self.keep_rate = None
        self.threshold_score = None

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths):
        keep_tokens = torch.round(sentence_lengths * self.keep_rate).long()
        device = attention_mask.device
        sz = attention_probs.shape[-1]
        batch_size = attention_mask.shape[0]
        self.threshold_score = torch.zeros((batch_size,))
        if self.keep_rate == 1:
            return attention_mask

        # compute the pruning scores by summing the attention probabilities over all heads
        attention_mask_index = (attention_mask < 0).permute(0, 1, 3, 2).repeat(1, attention_probs.shape[1], 1, sz)
        attention_probs[attention_mask_index] = 0
        pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)

        # sort the pruning scores using the top-k engine
        top_scores, sorted_indices = torch.sort(-pruning_scores, dim=-1)

        # construct the new attention mask
        new_attention_mask = torch.ones(attention_mask.shape, device=attention_mask.device) * -10000
        # TODO: remove for loop if possible
        for i in range(batch_size):
            new_attention_mask[i, ..., sorted_indices[i, 0:keep_tokens[i]]] = 0
            # if keep_tokens[i] < sz:
            #    self.threshold_score[i] = -top_scores[i, keep_tokens[i]] / torch.max(-top_scores[i, ...])

        return new_attention_mask


class ThresholdTokenPruner:
    """
    implements the layer-by-layer operations for threshold token pruning, where tokens are pruned if the importance
    score is strictly less than a given fraction of the maximum token importance score
    """
    def __init__(self, token_threshold):
        self.keep_threshold = token_threshold

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths):
        sz = attention_probs.shape[-1]
        batch_size = attention_mask.shape[0]
        if self.keep_threshold == 0:
            return attention_mask

        # compute the pruning scores by summing the attention probabilities over all heads
        attention_mask_index = (attention_mask < 0).permute(0, 1, 3, 2).repeat(1, attention_probs.shape[1], 1, sz)
        attention_probs[attention_mask_index] = 0
        pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)

        max_pruning_scores, _ = torch.max(pruning_scores, dim=-1, keepdim=True)
        relative_pruning_scores = pruning_scores / max_pruning_scores

        # construct the new attention mask
        new_attention_mask = torch.zeros(attention_mask.shape, device=attention_mask.device)
        new_attention_mask[relative_pruning_scores.unsqueeze(1).unsqueeze(1) < self.keep_threshold] = -10000

        print((new_attention_mask == 0).view(batch_size, -1).sum(dim=1).detach().cpu().numpy())
        return new_attention_mask


TOKEN_PRUNERS = {'topk': CascadeTokenPruner, 'threshold': ThresholdTokenPruner}

