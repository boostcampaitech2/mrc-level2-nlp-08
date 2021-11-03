# from transformers import (DPRContextEncoder,
#                           DPRContextEncoderTokenizer,
#                           DPRQuestionEncoder,
#                           DPRQuestionEncoderTokenizer)

# p_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
# q_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

from sentence_transformers import SentenceTransformer
import datasets
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

td = load_from_disk("../train_with_origin_gt_add_top_k_passage/not_include_answer_passage_train_es_top_4")
vd = load_from_disk("../valid_with_origin_gt_add_top_k_passage/not_include_answer_passage_valid_es_top_4")

model = SentenceTransformer("KR-SBERT/KR-SBERT-V40K-klueNLI-augSTS")

train_examples = [InputExample(texts=[td[i]["question"], td[i]["contexts"]]) for i in range(len(td))]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

valid_exampels = [InputExample(texts=[vd[i]["question"], vd[i]["contexts"]]) for i in range(len(vd))]
evaluator = evaluation.EmbeddingSimilarityEvaluator(valid_exampels, [1.0, 0.0, 0.0, 0.0, 0.0])

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    evaluator=evaluator,
    evaluation_steps=500,
)


# model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-single-nq-base")
# embeddings = model.encode(sentences)
# similarities = cosine_similarity(embeddings)
# print(embeddings)
# print(similarities)
