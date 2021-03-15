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
        self.cascading_attention_mask = None
        self.keep_tokens = None
        self.keep_rate = None
        self.threshold_score = None

    def update_attention_mask(self, attention_mask, attention_probs):
        sz = attention_probs.shape[-1]
        batch_size = attention_mask.shape[0]
        self.threshold_score = torch.zeros((batch_size,))
        if self.keep_rate == 1:
            self.cascading_attention_mask = attention_mask

            return attention_mask

        # compute the pruning scores by summing the attention probabilities over all heads
        pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)

        # sort the pruning scores using the top-k engine
        top_scores, sorted_indices = torch.sort(-pruning_scores, dim=-1)

        # construct the new attention mask
        new_attention_mask = torch.ones(attention_mask.shape, device=attention_mask.device) * -10000
        # TODO: remove for loop if possible
        for i in range(batch_size):
            new_attention_mask[i, ..., sorted_indices[i, 0:self.keep_tokens[i]]] = 0
            if self.keep_tokens[i] < sz:
                self.threshold_score[i] = -top_scores[i, self.keep_tokens[i]] / torch.max(-top_scores[i, ...])

        # update cascading attention_mask
        self.cascading_attention_mask = new_attention_mask

