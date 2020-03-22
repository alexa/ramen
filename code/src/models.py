# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
from transformers.modeling_roberta import RobertaLMHead
from transformers.modeling_bert import BertOnlyMLMHead


class BertMLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    """
    def __init__(self, config):
        super(BertMLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self.tie_weights()

    def get_trainable_parameters(self):
        # this is useful when freezing the encoder parameters
        return list(self.bert.embeddings.word_embeddings.parameters()) + [self.cls.predictions.bias]

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        pred_mask = masked_lm_labels.ne(-1)
        pred_vect = sequence_output[pred_mask]  # (bs, slen, dim)
        y = torch.masked_select(masked_lm_labels, pred_mask)
        prediction_scores = self.cls(pred_vect)
        masked_lm_loss = F.cross_entropy(prediction_scores, y)
        return masked_lm_loss


class RobertaMLM(BertPreTrainedModel):
    """RoBERTa model with the masked language modeling head.
    """
    def __init__(self, config):
        super(RobertaMLM, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()
        self.tie_weights()

    def get_trainable_parameters(self):
        return list(self.roberta.embeddings.word_embeddings.parameters()) + [self.lm_head.bias]

    def tie_weights(self):
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        outputs = self.roberta(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        pred_mask = masked_lm_labels.ne(-1)
        pred_vect = sequence_output[pred_mask]  # (bs, slen, dim)
        y = torch.masked_select(masked_lm_labels, pred_mask)
        prediction_scores = self.lm_head(pred_vect)
        masked_lm_loss = F.cross_entropy(prediction_scores, y)
        return masked_lm_loss


class BertAdaptor(nn.Module):
    """
    A class for adapting English BERT to other languages
    """
    def __init__(self, src_model, tgt_model):
        """
        src_model: (BertForMaskedLM) of English
        tgt_model: (BertForMaskedLM) of Foreign
        """
        super(BertAdaptor, self).__init__()
        self.src_model = src_model
        self.tgt_model = tgt_model

        # force sharing params
        self.tgt_model.bert.encoder = self.src_model.bert.encoder
        self.tgt_model.bert.pooler = self.src_model.bert.pooler
        # share embedding params
        self.tgt_model.bert.embeddings.position_embeddings = self.src_model.bert.embeddings.position_embeddings
        self.tgt_model.bert.embeddings.token_type_embeddings = self.src_model.bert.embeddings.token_type_embeddings
        self.tgt_model.bert.embeddings.LayerNorm = self.src_model.bert.embeddings.LayerNorm
        # share output layers
        self.tgt_model.cls.predictions.transform = self.src_model.cls.predictions.transform

    def forward(self, lang, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None):
        model = self.src_model if lang == 'en' else self.tgt_model
        return model(input_ids, token_type_ids, attention_mask=attention_mask,
                     masked_lm_labels=masked_lm_labels)


class RobertaAdaptor(nn.Module):
    """
    A class for adapting English BERT to other languages
    """
    def __init__(self, src_model, tgt_model):
        """
        src_model: (BertForMaskedLM) of English
        tgt_model: (Roberta) of Foreign
        """
        super(RobertaAdaptor, self).__init__()
        self.src_model = src_model
        self.tgt_model = tgt_model

        # force sharing params
        self.tgt_model.roberta.encoder = self.src_model.roberta.encoder
        self.tgt_model.roberta.pooler = self.src_model.roberta.pooler
        # share embedding params
        self.tgt_model.roberta.embeddings.position_embeddings = self.src_model.roberta.embeddings.position_embeddings
        self.tgt_model.roberta.embeddings.token_type_embeddings = self.src_model.roberta.embeddings.token_type_embeddings
        self.tgt_model.roberta.embeddings.LayerNorm = self.src_model.roberta.embeddings.LayerNorm
        # share output layers
        self.tgt_model.lm_head.dense = self.src_model.lm_head.dense
        self.tgt_model.lm_head.layer_norm = self.src_model.lm_head.layer_norm

    def forward(self, lang, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None):
        model = self.src_model if lang == 'en' else self.tgt_model
        return model(input_ids, token_type_ids, attention_mask=attention_mask,
                     masked_lm_labels=masked_lm_labels)
