import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules import dropout
from torch.nn.modules.conv import Conv1d

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    QuestionAnsweringModelOutput,
)
from transformers import AutoModel


class DoubleRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, pretrained_model_name_or_path, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        assert (
            "roberta" in config.model_type.lower()
        ), "Base model does not match with any Roberta variants"

        self.roberta1 = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            add_pooling_layer=False,
        )

        self.roberta2 = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            add_pooling_layer=False,
        )

        self.hidden_dim = config.hidden_size

        self.qa_outputs = nn.Linear(
            in_features=self.hidden_dim * 2, out_features=config.num_labels
        )

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

        outputs1 = self.roberta1(
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

        sequence_output1 = outputs1[0]

        outputs2 = self.roberta2(
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

        sequence_output2 = outputs2[0]
        # print(f"{sequence_output.shape=}")

        # logits = self.qa_outputs(sequence_output)
        # print(f"{logits.shape=}")

        concat_outputs = torch.cat((sequence_output1, sequence_output2), dim=-1)
        logits = self.qa_outputs(concat_outputs)

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
