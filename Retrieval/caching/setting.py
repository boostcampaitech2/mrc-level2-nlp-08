import pandas as pd
import pickle

def main():
    wiki_dataset = pd.read_json("/opt/ml/data/preprocess_wiki.json", orient="index")

    context_id_pair = {
        k: v for k, v in zip(wiki_dataset["text"], wiki_dataset["document_id"])
    }
    id_context_pair = {
        k: v for k, v in zip(wiki_dataset["document_id"], wiki_dataset["text"])
    }
    id_title_pair = {
        k: v for k, v in zip(wiki_dataset["document_id"], wiki_dataset["title"])
    }

    with open(
        "/opt/ml/mrc-level2-nlp-08/Retrieval/caching/wiki_context_id_pair.bin", "wb"
    ) as file:
        pickle.dump(context_id_pair, file)
    with open(
        "/opt/ml/mrc-level2-nlp-08/Retrieval/caching/wiki_id_context_pair.bin", "wb"
    ) as file:
        pickle.dump(id_context_pair, file)
    with open(
        "/opt/ml/mrc-level2-nlp-08/Retrieval/caching/id_title_pair.bin", "wb"
    ) as file:
        pickle.dump(id_title_pair, file)

    print("Caching Done")


if __name__ == "__main__":
    main()