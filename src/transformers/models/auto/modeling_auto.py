# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Model class. """


import warnings
from collections import OrderedDict

from ...utils import logging
from ..albert.modeling_albert import (
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
)
from ..bart.modeling_bart import (
    BartForCausalLM,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
)
from ..bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertModel,
)
from ..bert_generation.modeling_bert_generation import BertGenerationDecoder, BertGenerationEncoder
from ..big_bird.modeling_big_bird import (
    BigBirdForCausalLM,
    BigBirdForMaskedLM,
    BigBirdForMultipleChoice,
    BigBirdForPreTraining,
    BigBirdForQuestionAnswering,
    BigBirdForSequenceClassification,
    BigBirdForTokenClassification,
    BigBirdModel,
)
from ..blenderbot.modeling_blenderbot import BlenderbotForCausalLM, BlenderbotForConditionalGeneration, BlenderbotModel
from ..blenderbot_small.modeling_blenderbot_small import (
    BlenderbotSmallForCausalLM,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallModel,
)
from ..camembert.modeling_camembert import (
    CamembertForCausalLM,
    CamembertForMaskedLM,
    CamembertForMultipleChoice,
    CamembertForQuestionAnswering,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
)
from ..convbert.modeling_convbert import (
    ConvBertForMaskedLM,
    ConvBertForMultipleChoice,
    ConvBertForQuestionAnswering,
    ConvBertForSequenceClassification,
    ConvBertForTokenClassification,
    ConvBertModel,
)
from ..ctrl.modeling_ctrl import CTRLForSequenceClassification, CTRLLMHeadModel, CTRLModel
from ..deberta.modeling_deberta import (
    DebertaForMaskedLM,
    DebertaForQuestionAnswering,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    DebertaModel,
)
from ..deberta_v2.modeling_deberta_v2 import (
    DebertaV2ForMaskedLM,
    DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Model,
)
from ..distilbert.modeling_distilbert import (
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from ..dpr.modeling_dpr import DPRQuestionEncoder
from ..electra.modeling_electra import (
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
)
from ..encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from ..flaubert.modeling_flaubert import (
    FlaubertForMultipleChoice,
    FlaubertForQuestionAnsweringSimple,
    FlaubertForSequenceClassification,
    FlaubertForTokenClassification,
    FlaubertModel,
    FlaubertWithLMHeadModel,
)
from ..fsmt.modeling_fsmt import FSMTForConditionalGeneration, FSMTModel
from ..funnel.modeling_funnel import (
    FunnelForMaskedLM,
    FunnelForMultipleChoice,
    FunnelForPreTraining,
    FunnelForQuestionAnswering,
    FunnelForSequenceClassification,
    FunnelForTokenClassification,
    FunnelModel,
)
from ..gpt2.modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model

# Add modeling imports here
from ..gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoModel
from ..ibert.modeling_ibert import (
    IBertForMaskedLM,
    IBertForMultipleChoice,
    IBertForQuestionAnswering,
    IBertForSequenceClassification,
    IBertForTokenClassification,
    IBertModel,
)
from ..ltp.modeling_ltp import (
    LTPForMaskedLM,
    LTPForMultipleChoice,
    LTPForQuestionAnswering,
    LTPForSequenceClassification,
    LTPForTokenClassification,
    LTPModel,
)
from ..layoutlm.modeling_layoutlm import (
    LayoutLMForMaskedLM,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMModel,
)
from ..led.modeling_led import (
    LEDForConditionalGeneration,
    LEDForQuestionAnswering,
    LEDForSequenceClassification,
    LEDModel,
)
from ..longformer.modeling_longformer import (
    LongformerForMaskedLM,
    LongformerForMultipleChoice,
    LongformerForQuestionAnswering,
    LongformerForSequenceClassification,
    LongformerForTokenClassification,
    LongformerModel,
)
from ..lxmert.modeling_lxmert import LxmertForPreTraining, LxmertForQuestionAnswering, LxmertModel
from ..m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration, M2M100Model
from ..marian.modeling_marian import MarianForCausalLM, MarianModel, MarianMTModel
from ..mbart.modeling_mbart import (
    MBartForCausalLM,
    MBartForConditionalGeneration,
    MBartForQuestionAnswering,
    MBartForSequenceClassification,
    MBartModel,
)
from ..mobilebert.modeling_mobilebert import (
    MobileBertForMaskedLM,
    MobileBertForMultipleChoice,
    MobileBertForNextSentencePrediction,
    MobileBertForPreTraining,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertForTokenClassification,
    MobileBertModel,
)
from ..mpnet.modeling_mpnet import (
    MPNetForMaskedLM,
    MPNetForMultipleChoice,
    MPNetForQuestionAnswering,
    MPNetForSequenceClassification,
    MPNetForTokenClassification,
    MPNetModel,
)
from ..mt5.modeling_mt5 import MT5ForConditionalGeneration, MT5Model
from ..openai.modeling_openai import OpenAIGPTForSequenceClassification, OpenAIGPTLMHeadModel, OpenAIGPTModel
from ..pegasus.modeling_pegasus import PegasusForCausalLM, PegasusForConditionalGeneration, PegasusModel
from ..prophetnet.modeling_prophetnet import ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel
from ..rag.modeling_rag import (  # noqa: F401 - need to import all RagModels to be in globals() function
    RagModel,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)
from ..reformer.modeling_reformer import (
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerForSequenceClassification,
    ReformerModel,
    ReformerModelWithLMHead,
)
from ..retribert.modeling_retribert import RetriBertModel
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from ..speech_to_text.modeling_speech_to_text import Speech2TextForConditionalGeneration, Speech2TextModel
from ..squeezebert.modeling_squeezebert import (
    SqueezeBertForMaskedLM,
    SqueezeBertForMultipleChoice,
    SqueezeBertForQuestionAnswering,
    SqueezeBertForSequenceClassification,
    SqueezeBertForTokenClassification,
    SqueezeBertModel,
)
from ..t5.modeling_t5 import T5ForConditionalGeneration, T5Model
from ..tapas.modeling_tapas import (
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
)
from ..transfo_xl.modeling_transfo_xl import TransfoXLForSequenceClassification, TransfoXLLMHeadModel, TransfoXLModel
from ..vit.modeling_vit import ViTForImageClassification, ViTModel
from ..wav2vec2.modeling_wav2vec2 import Wav2Vec2ForMaskedLM, Wav2Vec2Model
from ..xlm.modeling_xlm import (
    XLMForMultipleChoice,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from ..xlm_prophetnet.modeling_xlm_prophetnet import (
    XLMProphetNetForCausalLM,
    XLMProphetNetForConditionalGeneration,
    XLMProphetNetModel,
)
from ..xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaForCausalLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from ..xlnet.modeling_xlnet import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnsweringSimple,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
)
from .auto_factory import auto_class_factory
from .configuration_auto import (
    AlbertConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BigBirdConfig,
    BlenderbotConfig,
    BlenderbotSmallConfig,
    CamembertConfig,
    ConvBertConfig,
    CTRLConfig,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    GPTNeoConfig,
    IBertConfig,
    LTPConfig,
    LayoutLMConfig,
    LEDConfig,
    LongformerConfig,
    LxmertConfig,
    M2M100Config,
    MarianConfig,
    MBartConfig,
    MobileBertConfig,
    MPNetConfig,
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    Speech2TextConfig,
    SqueezeBertConfig,
    T5Config,
    TapasConfig,
    TransfoXLConfig,
    ViTConfig,
    Wav2Vec2Config,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
)


logger = logging.get_logger(__name__)


MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (GPTNeoConfig, GPTNeoModel),
        (BigBirdConfig, BigBirdModel),
        (Speech2TextConfig, Speech2TextModel),
        (ViTConfig, ViTModel),
        (Wav2Vec2Config, Wav2Vec2Model),
        (M2M100Config, M2M100Model),
        (ConvBertConfig, ConvBertModel),
        (LEDConfig, LEDModel),
        (BlenderbotSmallConfig, BlenderbotSmallModel),
        (RetriBertConfig, RetriBertModel),
        (MT5Config, MT5Model),
        (T5Config, T5Model),
        (PegasusConfig, PegasusModel),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartModel),
        (BlenderbotConfig, BlenderbotModel),
        (DistilBertConfig, DistilBertModel),
        (AlbertConfig, AlbertModel),
        (CamembertConfig, CamembertModel),
        (XLMRobertaConfig, XLMRobertaModel),
        (BartConfig, BartModel),
        (LongformerConfig, LongformerModel),
        (RobertaConfig, RobertaModel),
        (LayoutLMConfig, LayoutLMModel),
        (SqueezeBertConfig, SqueezeBertModel),
        (BertConfig, BertModel),
        (OpenAIGPTConfig, OpenAIGPTModel),
        (GPT2Config, GPT2Model),
        (MobileBertConfig, MobileBertModel),
        (TransfoXLConfig, TransfoXLModel),
        (XLNetConfig, XLNetModel),
        (FlaubertConfig, FlaubertModel),
        (FSMTConfig, FSMTModel),
        (XLMConfig, XLMModel),
        (CTRLConfig, CTRLModel),
        (ElectraConfig, ElectraModel),
        (ReformerConfig, ReformerModel),
        (FunnelConfig, FunnelModel),
        (LxmertConfig, LxmertModel),
        (BertGenerationConfig, BertGenerationEncoder),
        (DebertaConfig, DebertaModel),
        (DebertaV2Config, DebertaV2Model),
        (DPRConfig, DPRQuestionEncoder),
        (XLMProphetNetConfig, XLMProphetNetModel),
        (ProphetNetConfig, ProphetNetModel),
        (MPNetConfig, MPNetModel),
        (TapasConfig, TapasModel),
        (MarianConfig, MarianModel),
        (IBertConfig, IBertModel),
        (LTPConfig, LTPModel),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (RetriBertConfig, RetriBertModel),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForPreTraining),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForPreTraining),
        (BigBirdConfig, BigBirdForPreTraining),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MobileBertConfig, MobileBertForPreTraining),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForPreTraining),
        (LxmertConfig, LxmertForPreTraining),
        (FunnelConfig, FunnelForPreTraining),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (LTPConfig, LTPForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        # Model with LM heads mapping
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForMaskedLM),
        (Speech2TextConfig, Speech2TextForConditionalGeneration),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (M2M100Config, M2M100ForConditionalGeneration),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (MarianConfig, MarianMTModel),
        (FSMTConfig, FSMTForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MobileBertConfig, MobileBertForMaskedLM),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (LTPConfig, LTPForMaskedLM),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForCausalLM),
        (CamembertConfig, CamembertForCausalLM),
        (XLMRobertaConfig, XLMRobertaForCausalLM),
        (RobertaConfig, RobertaForCausalLM),
        (BertConfig, BertLMHeadModel),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (
            XLMConfig,
            XLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, CTRLLMHeadModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (BertGenerationConfig, BertGenerationDecoder),
        (XLMProphetNetConfig, XLMProphetNetForCausalLM),
        (ProphetNetConfig, ProphetNetForCausalLM),
        (BartConfig, BartForCausalLM),
        (MBartConfig, MBartForCausalLM),
        (PegasusConfig, PegasusForCausalLM),
        (MarianConfig, MarianForCausalLM),
        (BlenderbotConfig, BlenderbotForCausalLM),
        (BlenderbotSmallConfig, BlenderbotSmallForCausalLM),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image Classification mapping
        (ViTConfig, ViTForImageClassification),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (BigBirdConfig, BigBirdForMaskedLM),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (MBartConfig, MBartForConditionalGeneration),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (ReformerConfig, ReformerForMaskedLM),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (LTPConfig, LTPForMaskedLM),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (M2M100Config, M2M100ForConditionalGeneration),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (MT5Config, MT5ForConditionalGeneration),
        (T5Config, T5ForConditionalGeneration),
        (PegasusConfig, PegasusForConditionalGeneration),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartForConditionalGeneration),
        (BlenderbotConfig, BlenderbotForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (XLMProphetNetConfig, XLMProphetNetForConditionalGeneration),
        (ProphetNetConfig, ProphetNetForConditionalGeneration),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (BigBirdConfig, BigBirdForSequenceClassification),
        (ConvBertConfig, ConvBertForSequenceClassification),
        (LEDConfig, LEDForSequenceClassification),
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (MBartConfig, MBartForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (LongformerConfig, LongformerForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (SqueezeBertConfig, SqueezeBertForSequenceClassification),
        (LayoutLMConfig, LayoutLMForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (MobileBertConfig, MobileBertForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
        (ElectraConfig, ElectraForSequenceClassification),
        (FunnelConfig, FunnelForSequenceClassification),
        (DebertaConfig, DebertaForSequenceClassification),
        (DebertaV2Config, DebertaV2ForSequenceClassification),
        (GPT2Config, GPT2ForSequenceClassification),
        (OpenAIGPTConfig, OpenAIGPTForSequenceClassification),
        (ReformerConfig, ReformerForSequenceClassification),
        (CTRLConfig, CTRLForSequenceClassification),
        (TransfoXLConfig, TransfoXLForSequenceClassification),
        (MPNetConfig, MPNetForSequenceClassification),
        (TapasConfig, TapasForSequenceClassification),
        (IBertConfig, IBertForSequenceClassification),
        (LTPConfig, LTPForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (BigBirdConfig, BigBirdForQuestionAnswering),
        (ConvBertConfig, ConvBertForQuestionAnswering),
        (LEDConfig, LEDForQuestionAnswering),
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (CamembertConfig, CamembertForQuestionAnswering),
        (BartConfig, BartForQuestionAnswering),
        (MBartConfig, MBartForQuestionAnswering),
        (LongformerConfig, LongformerForQuestionAnswering),
        (XLMRobertaConfig, XLMRobertaForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (SqueezeBertConfig, SqueezeBertForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnsweringSimple),
        (FlaubertConfig, FlaubertForQuestionAnsweringSimple),
        (MobileBertConfig, MobileBertForQuestionAnswering),
        (XLMConfig, XLMForQuestionAnsweringSimple),
        (ElectraConfig, ElectraForQuestionAnswering),
        (ReformerConfig, ReformerForQuestionAnswering),
        (FunnelConfig, FunnelForQuestionAnswering),
        (LxmertConfig, LxmertForQuestionAnswering),
        (MPNetConfig, MPNetForQuestionAnswering),
        (DebertaConfig, DebertaForQuestionAnswering),
        (DebertaV2Config, DebertaV2ForQuestionAnswering),
        (IBertConfig, IBertForQuestionAnswering),
        (LTPConfig, LTPForQuestionAnswering),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Table Question Answering mapping
        (TapasConfig, TapasForQuestionAnswering),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (BigBirdConfig, BigBirdForTokenClassification),
        (ConvBertConfig, ConvBertForTokenClassification),
        (LayoutLMConfig, LayoutLMForTokenClassification),
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (LongformerConfig, LongformerForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (SqueezeBertConfig, SqueezeBertForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (MobileBertConfig, MobileBertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (FunnelConfig, FunnelForTokenClassification),
        (MPNetConfig, MPNetForTokenClassification),
        (DebertaConfig, DebertaForTokenClassification),
        (DebertaV2Config, DebertaV2ForTokenClassification),
        (IBertConfig, IBertForTokenClassification),
        (LTPConfig, LTPForTokenClassification),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (BigBirdConfig, BigBirdForMultipleChoice),
        (ConvBertConfig, ConvBertForMultipleChoice),
        (CamembertConfig, CamembertForMultipleChoice),
        (ElectraConfig, ElectraForMultipleChoice),
        (XLMRobertaConfig, XLMRobertaForMultipleChoice),
        (LongformerConfig, LongformerForMultipleChoice),
        (RobertaConfig, RobertaForMultipleChoice),
        (SqueezeBertConfig, SqueezeBertForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (DistilBertConfig, DistilBertForMultipleChoice),
        (MobileBertConfig, MobileBertForMultipleChoice),
        (XLNetConfig, XLNetForMultipleChoice),
        (AlbertConfig, AlbertForMultipleChoice),
        (XLMConfig, XLMForMultipleChoice),
        (FlaubertConfig, FlaubertForMultipleChoice),
        (FunnelConfig, FunnelForMultipleChoice),
        (MPNetConfig, MPNetForMultipleChoice),
        (IBertConfig, IBertForMultipleChoice),
        (LTPConfig, LTPForMultipleChoice),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, BertForNextSentencePrediction),
        (MobileBertConfig, MobileBertForNextSentencePrediction),
    ]
)


AutoModel = auto_class_factory("AutoModel", MODEL_MAPPING)

AutoModelForPreTraining = auto_class_factory(
    "AutoModelForPreTraining", MODEL_FOR_PRETRAINING_MAPPING, head_doc="pretraining"
)

# Private on puprose, the public class will add the deprecation warnings.
_AutoModelWithLMHead = auto_class_factory(
    "AutoModelWithLMHead", MODEL_WITH_LM_HEAD_MAPPING, head_doc="language modeling"
)

AutoModelForCausalLM = auto_class_factory(
    "AutoModelForCausalLM", MODEL_FOR_CAUSAL_LM_MAPPING, head_doc="causal language modeling"
)

AutoModelForMaskedLM = auto_class_factory(
    "AutoModelForMaskedLM", MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked language modeling"
)

AutoModelForSeq2SeqLM = auto_class_factory(
    "AutoModelForSeq2SeqLM",
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="t5-base",
)

AutoModelForSequenceClassification = auto_class_factory(
    "AutoModelForSequenceClassification", MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, head_doc="sequence classification"
)

AutoModelForQuestionAnswering = auto_class_factory(
    "AutoModelForQuestionAnswering", MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

AutoModelForTableQuestionAnswering = auto_class_factory(
    "AutoModelForTableQuestionAnswering",
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)

AutoModelForNextSentencePrediction = auto_class_factory(
    "AutoModelForNextSentencePrediction",
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    head_doc="next sentence prediction",
)

AutoModelForImageClassification = auto_class_factory(
    "AutoModelForImageClassification", MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, head_doc="image classification"
)


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
