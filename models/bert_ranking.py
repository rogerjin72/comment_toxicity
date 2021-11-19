import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import KLDivLoss
from transformers import BertPreTrainedModel, BertModel


class BertForRanking(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        comparison_input_ids=None,
        comparison_attention_mask=None,
        comparison_token_type_ids=None,
        values=None,
        output_attentions=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        first_seq = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        first_seq_hidden = first_seq[1]
        first_seq_hidden = self.dropout(first_seq_hidden)
        first_seq_score = self.classifier(first_seq_hidden)

        if comparison_input_ids is None:
            return first_seq_score

        second_seq = self.bert(
            comparison_input_ids,
            attention_mask=comparison_attention_mask,
            token_type_ids=comparison_token_type_ids
        )

        second_seq_hidden = second_seq[1]
        second_seq_hidden = self.dropout(second_seq_hidden)
        second_seq_score = self.classifier(second_seq_hidden)
        outputs = torch.cat((first_seq_score, second_seq_score), -1)

        logits = F.log_softmax(outputs, dim=-1)
        
        if values is not None:
            try:
                prob = np.array([values])
            except TypeError:
                prob = values.cpu().numpy()
                prob = prob.reshape(1, len(prob))
                print(prob.shape)

            prob = np.concatenate([1-prob, prob]).T
            prob = torch.Tensor(prob)

            f = KLDivLoss()
            loss = f(logits, prob)

        return loss, outputs
