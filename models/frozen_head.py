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
    ROBERTA_START_DOCSTRING,
    ROBERTA_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
)
from transformers import AutoModel


@add_start_docstrings(
    """
    Roberta Model with a LSTM span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class CustomModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, pretrained_model_name_or_path, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        assert "roberta" in config.model_type.lower(), "Base model does not match with any Roberta variants"

        self.roberta = AutoModel.from_pretrained(
            "/home/develop/torch_versionup/checkpoint-666",
            config=config,
            add_pooling_layer=False,
        )

        for p in self.roberta.parameters():
            p.requires_grad = False

        self.hidden_dim = config.hidden_size
        # self.qa_outputs = nn.Linear(in_features=self.hidden_dim, out_features=config.num_labels)

        # self.qa_outputs = nn.Linear(in_features=1024 * 2, out_features=config.num_labels)
        self.qa_outputs = nn.Linear(in_features=256 * 3, out_features=config.num_labels)

        # self.conv1d_k1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=256, kernel_size=1, padding=0)
        # self.conv1d_k3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=256, kernel_size=3, padding=1)
        # self.conv1d_k5 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=256, kernel_size=5, padding=2)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=1024 * 2, out_features=256)
        self.fc2 = nn.Linear(in_features=512 * 2, out_features=256)
        self.fc3 = nn.Linear(in_features=256 * 2, out_features=256)
        self.gru1 = nn.GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.gru3 = nn.GRU(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        # self.lstm1 = nn.LSTM(
        #     input_size=1024,
        #     hidden_size=1024,
        #     num_layers=2,
        #     dropout=0.5,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.lstm2 = nn.LSTM(
        #     input_size=1024,
        #     hidden_size=512,
        #     num_layers=2,
        #     dropout=0.5,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.lstm3 = nn.LSTM(
        #     input_size=1024,
        #     hidden_size=256,
        #     num_layers=2,
        #     dropout=0.5,
        #     batch_first=True,
        #     bidirectional=True,
        # )

        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.hidden_size, nhead=4, dim_feedforward=4096, activation="gelu", batch_first=True
        # )
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
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
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        # print(f"{sequence_output.shape=}")

        # logits = self.qa_outputs(sequence_output)
        # print(f"{logits.shape=}")

        # sequence_output = outputs[0].permute(0, 2, 1)
        # # print(f"{sequence_output.shape=}")

        # cnn_k1_output = self.relu(self.conv1d_k1(sequence_output))
        # cnn_k3_output = self.relu(self.conv1d_k3(sequence_output))
        # cnn_k5_output = self.relu(self.conv1d_k5(sequence_output))
        # concat_cnn_output = torch.cat((cnn_k1_output, cnn_k3_output, cnn_k5_output), 1)

        # # logits = self.qa_outputs(self.dropout(concat_cnn_output.permute(0, 2, 1)))
        # # print(concat_cnn_output.shape)
        # lstm_output, (c, h) = self.lstm(concat_cnn_output.permute(0, 2, 1))
        # logits = self.qa_outputs(lstm_output)

        gru_output1, (n_h) = self.gru1(sequence_output)
        gru_output2, (n_h) = self.gru2(sequence_output)
        gru_output3, (n_h) = self.gru3(sequence_output)

        gru_output1 = self.fc1(gru_output1)
        gru_output2 = self.fc2(gru_output2)
        gru_output3 = self.fc3(gru_output3)

        gru_output = torch.cat((gru_output1, gru_output2, gru_output3), dim=-1)

        logits = self.qa_outputs(gru_output)

        # lstm_output1, (c1, h1) = self.lstm1(sequence_output)
        # lstm_output2, (c2, h2) = self.lstm2(sequence_output)
        # lstm_output3, (c3, h3) = self.lstm3(sequence_output)

        # lstm_output1 = self.fc1(lstm_output1)
        # lstm_output2 = self.fc2(lstm_output2)
        # lstm_output3 = self.fc3(lstm_output3)

        # lstm_output = torch.cat((lstm_output1, lstm_output2, lstm_output3), dim=-1)

        # logits = self.qa_outputs(lstm_output)

        # transformer_output = self.encoder(sequence_output)
        # logits = self.qa_outputs(transformer_output)

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
