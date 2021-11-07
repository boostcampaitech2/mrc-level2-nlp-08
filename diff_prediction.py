import json
from datasets import load_from_disk

with open("/opt/ml/develop/baseline_inference/sota.json", encoding="utf-8") as json_file:
    json1 = json.load(json_file)
with open(
    "/opt/ml/develop/baseline_inference/prediction.json",
    encoding="utf-8",
) as json_file:
    json2 = json.load(json_file)

mrc_id = list(json1.keys())

count = 0
for i in range(len(mrc_id)):
    if json1[mrc_id[i]] != json2[mrc_id[i]]:
        count += 1
print(f"sota <-> soft_vot 다른개수: {count}")
