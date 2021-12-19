import torch

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List

@dataclass
class LTPEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None
    ffn_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None


@dataclass
class LTPSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None
    ffn_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None


@dataclass
class LTPModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None
    ffn_sentence_lengths: Optional[List[List[torch.FloatTensor]]] = None
