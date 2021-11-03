from pororo import Pororo
from tqdm import tqdm
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "true"

c = pd.read_csv(
    "/home/qa_generation/preprocess_and_except_one_char_answer_ner_dataset.csv"
)

qg = Pororo(task="qg", lang="ko")
print("start")
n = qg(list(c["title"]), list(c["text"]))

if len(n) == len(c):
    c["question"] = n
    c.to_csv("hoow.csv")
else:
    h = pd.DataFrame(data={"question": n})
    h.to_csv("hhaa.csv")
