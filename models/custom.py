import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules import dropout
from torch.nn.modules.conv import Conv1d

import numpy as np

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
import math
from sentence_transformers import SentenceTransformer, util
import pickle


@add_start_docstrings(
    """
    Roberta Model with a LSTM span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class MyModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, pretrained_model_name_or_path, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = tokenizer

        assert "roberta" in config.model_type.lower(), "Base model does not match with any Roberta variants"

        self.sroberta = SentenceTransformer("Huffon/sentence-klue-roberta-base")
        with open("passage_embedding.pickle", "rb") as f:
            self.passage_embedding = pickle.load(f)

        self.roberta = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=config, add_pooling_layer=False
        )
        self.hidden_dim = config.hidden_size

        # self.qa_outputs = nn.Linear(in_features=64 * 3, out_features=config.num_labels)
        self.qa_outputs = nn.Linear(in_features=768, out_features=config.num_labels)
        self.question_linear_layer = nn.Linear(in_features=64 * self.hidden_dim, out_features=768)
        self.passage_linear_layer = nn.Linear(in_features=self.hidden_dim, out_features=768)
        self.sentence_sequence_linear_layer = nn.Linear(in_features=self.hidden_dim, out_features=768)

        # self.conv1d_k1 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=1, padding=0)
        # self.conv1d_k3 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3, padding=1)
        # self.conv1d_k5 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

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
        # query_passage_embedding=None,
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

        # if query_passage_embedding is not None:
        _token_type_ids = self.get_token_type_ids(input_ids)

        question_idx = input_ids.index(self.tokenizer.sep_token_id)
        question_embedding = self.sroberta.encode(self.tokenizer.decode(input_ids[:question_idx]))

        # [batch=16, max_seq_len=384, hidden_dim=1024]
        question_vectors = self.relu(sequence_output * (_token_type_ids.unsqueeze(dim=-1) == 0))
        question_vectors = question_vectors[:, :64, :]  # query max len: 62, mean: 29.555
        question_vectors = question_vectors.view((sequence_output.shape[0], 1, -1))  # [B, 1, 64 * 1024]
        question_vectors = self.question_linear_layer(question_vectors)  # [batch=16, 1, hidden_dim=768]

        passage_vectors = self.relu(sequence_output * (_token_type_ids.unsqueeze(dim=-1) == 1))
        # [batch=16, max_seq_len=384, hidden_dim=1024]
        passage_vectors = self.passage_linear_layer(passage_vectors)
        # [batch=16, max_seq_len=384, hidden_dim=768]

        # TODO: Average pooling, maxpooling

        # print(query_passage_embedding.shape)  # [batch=16, (q, p)=2, s_roberta_hidden_dim=768]
        # sentence_query_embedding = query_passage_embedding[:, 0, :]  # [batch=16, s_roberta_hidden_dim=768]
        # sts = util.pytorch_cos_sim(sentence_query_embedding, query_passage_embedding)  # [batch, 768]

        # question_vectors = (
        #     question_vectors + sentence_query_embedding[-1]
        # )  # [batch=16, question_len=64, hidden_dim=768]

        # sentence_passage_embedding = query_passage_embedding[:, 1, :]  # [batch=16, s_roberta_hidden_dim=768]
        # passage_vectors = (
        #     passage_vectors + sentence_passage_embedding[-1]
        # )  # [batch=16, question_len=384, hidden_dim=768]

        # cnn_input = sequence_output.permute(0, 2, 1)
        # cnn_input = sequence_output.permute(0, 2, 1)
        # # print(f"{sequence_output.shape=}")

        # cnn_k1_output = self.relu(self.conv1d_k1(cnn_input))
        # cnn_k3_output = self.relu(self.conv1d_k3(cnn_input))
        # cnn_k5_output = self.relu(self.conv1d_k5(cnn_input))
        # concat_cnn_output = torch.cat((cnn_k1_output, cnn_k3_output, cnn_k5_output), 1)

        # logits = self.qa_outputs(self.dropout(concat_cnn_output.permute(0, 2, 1)))
        # print(f"{logits.shape=}")

        question_passage_attention = torch.matmul(
            passage_vectors, question_vectors.permute(0, 2, 1)
        )  # [16, 384, 1]
        question_passage_attention = question_passage_attention / math.sqrt(
            passage_vectors.shape[-1]
        )  # [16, 384, 1]
        question_passage_attention = nn.functional.softmax(question_passage_attention, 1)  # [16, 384, 1]
        sentence_sequence_output = (
            self.sentence_sequence_linear_layer(sequence_output) * question_passage_attention
        )  # [16, 384, 768]

        logits = self.qa_outputs(sentence_sequence_output)  # [16, 384, 2]
        # else:
        #     logits = self.qa_outputs(self.sentence_sequence_linear_layer(sequence_output))

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

    def get_token_type_ids(self, input_ids):
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_index = np.where(input_id.cpu().numpy() == self.tokenizer.sep_token_id)
            token_type_id = [0] * sep_index[0][0] + [1] * (len(input_id) - sep_index[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
