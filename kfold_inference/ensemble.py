import os
import json
from collections import defaultdict


# files = os.listdir("please")

files = [
    "0_fold_infer",
    "1_fold_infer",
    "2_fold_infer",
    "3_fold_infer",
    "4_fold_infer",
]

files = ["please/" + f + "/nbest_predictions.json" for f in files]

json_files = []
for file in files:
    with open(f"{file}") as f:
        json_file = json.load(f)
        json_files.append(json_file)

k = 20
mrc_id = list(json_files[0].keys())

last_answer = {}
for i in range(len(mrc_id)):
    answer_dict = {}
    for j in range(k):

        for file in json_files:

            answer = file[mrc_id[i]][j]["text"]
            prob = file[mrc_id[i]][j]["probability"]

            if not answer in answer_dict:
                answer_dict[answer] = prob
            else:
                answer_dict[answer] += prob
    temp = list(answer_dict.items())
    temp.sort(key=lambda x: x[1], reverse=True)
    last_answer[mrc_id[i]] = temp[0][0]

with open(
    "please/hard_voting_predictions.json",
    "w",
    encoding="utf-8",
) as make_file:
    json.dump(last_answer, make_file, indent="\t", ensure_ascii=False)
