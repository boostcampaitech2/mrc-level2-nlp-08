from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk

td = load_from_disk("../train_with_origin_gt_add_top_k_passage/not_include_answer_passage_train_es_top_4")
sentence_0 = td[0]["question"]
sentence_1 = td[0]["context"]

model = SentenceTransformer("bespin-global/klue-korsts-roberta-base-sentence-embedding")
model.max_seq_length = 384

embeddings = model.encode([sentence_0, sentence_1])
print(embeddings.shape)
cos_sim = util.cos_sim(embeddings[0], embeddings[1])

print(cos_sim)
print(cos_sim.shape)
