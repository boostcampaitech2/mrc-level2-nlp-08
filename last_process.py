import json

with open(
    "/opt/ml/develop/ttttt/passage_random_shuffle_hb_v5/nbest_predictions.json", encoding="utf-8"
) as json_file:
    json_data = json.load(json_file)
with open(
    "/opt/ml/develop/ttttt/passage_random_shuffle_hb_v5/predictions.json", encoding="utf-8"
) as json_file:
    raw_answer = json.load(json_file)

mrc_id = list(json_data.keys())

last_answer = {}
for i in range(len(mrc_id)):
    answer_dict = {}
    for j in range(len(json_data[mrc_id[i]])):
        answer = json_data[mrc_id[i]][j]["text"]
        prob = json_data[mrc_id[i]][j]["probability"]
        if not answer in answer_dict:
            answer_dict[answer] = prob
        else:
            answer_dict[answer] += prob
    temp = list(answer_dict.items())
    temp.sort(key=lambda x: x[1], reverse=True)
    last_answer[mrc_id[i]] = temp[0][0]

for i in range(len(mrc_id)):
    if last_answer[mrc_id[i]] != raw_answer[mrc_id[i]]:
        print(mrc_id[i], ": ", raw_answer[mrc_id[i]], " -> ", last_answer[mrc_id[i]])

with open(
    "/opt/ml/develop/ttttt/passage_random_shuffle_hb_v5/proprecessing_predictions.json",
    "w",
    encoding="utf-8",
) as make_file:
    json.dump(last_answer, make_file, indent="\t", ensure_ascii=False)
