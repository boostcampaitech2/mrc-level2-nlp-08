from typing import Any, Tuple
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch.nn import CrossEntropyLoss
import math


class RobertModelForQuestionAnsweringConv(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()
        self.sep_token_id = config.sep_token_id
        self.query_drop_out = nn.Dropout(0.1)
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.conv1d_layer1 = nn.Conv1d(config.hidden_size, 1024, kernel_size=1)
        self.conv1d_layer3 = nn.Conv1d(
            config.hidden_size, 1024, kernel_size=3, padding=1
        )
        self.conv1d_layer5 = nn.Conv1d(
            config.hidden_size, 1024, kernel_size=5, padding=2
        )
        self.drop_out = nn.Dropout(0.3)
        self.classify_layer = nn.Linear(1024 * 3, 2, bias=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # if not token_type_ids:
        token_type_ids = self.make_token_type_ids(input_ids)
        embedded_query = sequence_output * (token_type_ids.unsqueeze(dim=-1) == 0)
        # 전체 Text 중 query에 해당하는 Embedded Vector만 남김.
        embedded_query = self.query_drop_out(F.relu(embedded_query))
        # Activation Function 및 Dropout Layer 통과
        embedded_query = self.query_layer(embedded_query)
        # Dense Layer를 통과 시킴. (B * max_seq_length * hidden_size)
        embedded_query = torch.mean(embedded_query, 1, keepdim=True)
        # Query에 해당하는 Token Embedding을 평균냄. (B * 1 * hidden_size)

        embedded_key = sequence_output * (token_type_ids.unsqueeze(dim=-1) == 1)
        # 전체 Text 중 context에 해당하는 Embedded Vector만 남김.
        embedded_key = self.key_layer(embedded_key)
        # (B * max_seq_length * hidden_size)
        embedded_value = self.value_layer(sequence_output)
        # (B * max_seq_length * hidden_size)
        attention_rate = torch.matmul(
            embedded_key, torch.transpose(embedded_query, 1, 2)
        )
        # Context의 Value Vector와 Quetion의 Query Vector를 사용 (B * max_seq_legth * 1)
        attention_rate = attention_rate / math.sqrt(embedded_key.shape[-1])
        # hidden size의 표준편차로 나눠줌. (B * max_seq_legth * 1)
        attention_rate = attention_rate / 10
        # Temperature로 나눠줌. (B * max_seq_legth * 1)
        attention_rate = F.softmax(attention_rate, 1)
        # softmax를 통과시켜서 확률값으로 변경해, Question과 Context의 Attention Rate를 구함. (B * max_seq_legth * 1)
        embedded_value = embedded_value * attention_rate
        # Attention Rate를 활용해서 Output 값을 변경함. (B * max_seq_legth * hidden_size)

        conv_input = embedded_value.transpose(1, 2)
        # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_output1 = F.relu(self.conv1d_layer1(conv_input))
        # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv1d_layer3(conv_input))
        # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv1d_layer5(conv_input))
        # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1)
        # Concatenation (B * num_conv_filter x 3 * max_seq_legth)

        concat_output = concat_output.transpose(1, 2)
        # Dense Layer에 입력을 위해 Transpose (B * max_seq_legth * num_conv_filter x 3)
        concat_output = self.drop_out(concat_output)
        # dropout 통과
        logits = self.classify_layer(concat_output)
        # Classifier Layer를 통해 최종 Logit을 얻음. (B * max_seq_legth * 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def make_token_type_ids(self, input_ids):
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)
            token_type_id = [0] * sep_idx[0][0] + [1] * (len(input_id) - sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()
