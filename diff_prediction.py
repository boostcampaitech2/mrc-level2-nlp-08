import json
from datasets import load_from_disk

with open("ttttt/hb_k_100_v3/hard_voting_predictions_total.json", encoding="utf-8") as json_file:
    json1 = json.load(json_file)
with open("ttttt/final_nbest/soft_voting_4.json", encoding="utf-8") as json_file:
    json2 = json.load(json_file)
# with open("ttttt/final_nbest/hard_voting.json", encoding="utf-8") as json_file:
with open(
    "/opt/ml/develop/ttttt/concat_train_gt_1_checkpoint-670/predictions.json", encoding="utf-8"
) as json_file:
    json3 = json.load(json_file)

test_dataset = load_from_disk("/opt/ml/data/test_dataset/validation")

id_question_table = {}
for _, row in enumerate(test_dataset):
    id_question_table[row["id"]] = row["question"]

mrc_id = list(json1.keys())

# count = 0
# for i in range(len(mrc_id)):
#     if json1[mrc_id[i]] != json2[mrc_id[i]]:
#         # print(f"{id_question_table[mrc_id[i]]} :  {json1[mrc_id[i]]} <->  {json2[mrc_id[i]]}")
#         count += 1
# print(f"sota <-> soft_vot 다른개수: {count}")

count = 0
for i in range(len(mrc_id)):
    if json1[mrc_id[i]] != json3[mrc_id[i]]:
        print(f"{id_question_table[mrc_id[i]]} :  {json1[mrc_id[i]]} <->  {json3[mrc_id[i]]}")
        count += 1
print(f"sota <-> hard_vot 다른개수: {count}")

# count = 0
# for i in range(len(mrc_id)):
#     if json2[mrc_id[i]] != json3[mrc_id[i]]:
#         # print(f"{id_question_table[mrc_id[i]]} :  {json2[mrc_id[i]]} <->  {json3[mrc_id[i]]}")
#         count += 1


# print(f"soft_vot <-> hard_vot 다른개수: {count}")
