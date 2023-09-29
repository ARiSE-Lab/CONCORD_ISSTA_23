# coding=utf-8

import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.distributed as dist

from transformers.activations import gelu, gelu_new
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import (
    BertOnlyMLMHead,
    BertPreTrainedModel,
    BertModel,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)



logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = []


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm

class ConcordTreePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.tree_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.tree_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class ConcordForContrastivePreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)  # lm_head
        self.tree_pred_head = ConcordTreePredictionHead(config)
        self.sim = nn.CosineSimilarity(dim=-1)
        # temperature for Contrastive Loss
        self.temp = config.temp
        self.emb_pooler_type = config.emb_pooler_type
        self.mlm_weight = config.mlm_weight
        self.clr_weight = config.clr_weight
        self.tree_pred_weight = config.tree_pred_weight
        assert self.emb_pooler_type in ["cls", "avg"]

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        mlm_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mlm_labels=None,
        tree_labels=None,
        mlm_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert attention_mask is not None and mlm_attention_mask is not None
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1) 
        # batch looks like [[A, A+, (A-)], [B, B+, (B-)]]
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        # input_ids inputs look like [A, A+, A-, B, B+, B-]
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]  # token rep (bs * num_sent, seq_len, hidden), [CLS] rep (bs * num_sent, hidden)

        if self.emb_pooler_type == 'avg':
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        pooled_output = pooled_output.view((batch_size, num_sent, pooled_output.size(-1)))  # (bs * num_sent, hidden) -> # (bs, num_sent, hidden)

        # orig_h is the original samples [A, B], clone_h is positive counterparts [A+, B+]
        orig_h, clone_h = pooled_output[:, 0], pooled_output[:, 1]  # (bs, hidden)
        deviant_h = pooled_output[:, 2]

        if dist.is_initialized():
            deviant_h_list = [torch.zeros_like(deviant_h) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=deviant_h_list, tensor=deviant_h.contiguous())
            deviant_h_list[dist.get_rank()] = deviant_h
            deviant_h = torch.cat(deviant_h_list, 0)

            orig_h_list = [torch.zeros_like(orig_h) for _ in range(dist.get_world_size())]
            clone_h_list = [torch.zeros_like(clone_h) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=orig_h_list, tensor=orig_h.contiguous())
            dist.all_gather(tensor_list=clone_h_list, tensor=clone_h.contiguous())
            orig_h_list[dist.get_rank()] = orig_h
            clone_h_list[dist.get_rank()] = clone_h
            # Get full batch embeddings: (bs x N, hidden)
            orig_h = torch.cat(orig_h_list, 0)
            clone_h = torch.cat(clone_h_list, 0)

        # similarity between original and clone embeddings
        orig_h_clone_h_cos_sim = self.sim(orig_h.unsqueeze(1), clone_h.unsqueeze(0)) / self.temp
        # similarity with Clone Deviant
        orig_h_deviant_h_cos = self.sim(orig_h.unsqueeze(1), deviant_h.unsqueeze(0)) / self.temp  # (bs, bs)
        cos_sim = torch.cat([orig_h_clone_h_cos_sim, orig_h_deviant_h_cos], 1)
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        deviant_h_weight = self.config.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - orig_h_deviant_h_cos.size(-1)) + [0.0] * i + [deviant_h_weight] + [0.0] * (
                        orig_h_deviant_h_cos.size(-1) - i - 1) for i in range(orig_h_deviant_h_cos.size(-1))]
        ).to(self.device)
        cos_sim = cos_sim + weights
        clr_loss = loss_fct(cos_sim, labels)
        total_loss = clr_loss
        output = outputs[:2]

        if mlm_input_ids is not None and mlm_labels is not None and tree_labels is not None:
            mlm_outputs = self.bert(
                mlm_input_ids,
                attention_mask=mlm_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            mlm_sequence_output = mlm_outputs[0]
            prediction_scores = self.cls(mlm_sequence_output)
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

            tree_prediction_scores = self.tree_pred_head(mlm_sequence_output)
            tree_pred_loss = loss_fct(tree_prediction_scores.view(-1, self.config.tree_vocab_size), tree_labels.view(-1))
            total_loss = self.clr_weight * clr_loss + self.mlm_weight * mlm_loss + self.tree_pred_weight * tree_pred_loss

        output = (total_loss, mlm_loss, clr_loss, tree_pred_loss) + output  # add hidden states and attention if they are here

        return output



class ConcordForEncoder(BertPreTrainedModel):
    def __init__(self, config, new_pooler=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.emb_pooler_type = config.emb_pooler_type
        self.pooler = BertPooler(config) if new_pooler else None
        assert self.emb_pooler_type in ["cls", "avg"]

        self.init_weights()


    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert attention_mask is not None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]  # token rep (bs * num_sent, seq_len, hidden), [CLS] rep (bs * num_sent, hidden)
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)

        if self.emb_pooler_type == 'avg':
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        return pooled_output  # (loss, mlm_loss, clr_loss), (sequence_output), (pooled_output)

class ConcordForCls(BertPreTrainedModel):
    def __init__(self, config, new_pooler):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pooler = BertPooler(config) if new_pooler else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.emb_pooler_type = config.emb_pooler_type

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert attention_mask is not None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        if self.emb_pooler_type == "avg":
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.output_proj(pooled_output)
        outputs = (logits,) + outputs[2:]

        assert labels is not None

        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        outputs = (cls_loss,) + outputs

        return outputs
