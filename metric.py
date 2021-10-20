from collections import defaultdict
from os import path
import json

from datasets import load_metric
from transformers.trainer_utils import EvalPrediction

import numpy as np
import re


def check_empty(prediction: list):
    if prediction:
        return prediction
    return [{"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}]


def compute_metrics(args, outputs: EvalPrediction):
    max_answer_length = args.max_answer_length
    num_max_prediction = args.num_max_prediction
    dataset = args.dataset["validation"]

    logits_of_start_idxs_predictions, logits_of_end_idxs_predictions = outputs.predictions
    prediction_cadidates_info = defaultdict(list)
    for (
        token_type_ids,
        offset_mapping,
        overflow_to_sample_mapping,
        logits_of_start_idxs_prediction,
        logits_of_end_idxs_prediction,
    ) in zip(
        args.token_type_ids,
        args.processed_eval_dataset["offset_mapping"],
        args.processed_eval_dataset["overflow_to_sample_mapping"],
        logits_of_start_idxs_predictions,
        logits_of_end_idxs_predictions,
    ):
        id = dataset["id"][overflow_to_sample_mapping]
        context = dataset["context"][overflow_to_sample_mapping]

        start_offset_idxs = np.argsort(logits_of_start_idxs_prediction)[-1 : -num_max_prediction - 1 : -1]
        end_offset_idxs = np.argsort(logits_of_end_idxs_prediction)[-1 : -num_max_prediction - 1 : -1]
        for start_offset_idx in start_offset_idxs:
            for end_offset_idx in end_offset_idxs:
                if (
                    start_offset_idx < len(offset_mapping)
                    and token_type_ids[start_offset_idx] == 1
                    and start_offset_idx <= end_offset_idx < len(offset_mapping)
                    and end_offset_idx - start_offset_idx <= max_answer_length
                    and offset_mapping[start_offset_idx][0] < offset_mapping[end_offset_idx][1]
                ):
                    prediction_cadidates_info[id].append(
                        {
                            "text": context[
                                offset_mapping[start_offset_idx][0] : offset_mapping[end_offset_idx][1]
                            ],
                            "start_logit": logits_of_start_idxs_prediction[start_offset_idx],
                            "end_logit": logits_of_end_idxs_prediction[end_offset_idx],
                            "score": logits_of_start_idxs_prediction[start_offset_idx]
                            + logits_of_end_idxs_prediction[end_offset_idx],
                        }
                    )

    predictions_info_per_id = {
        id: sorted(check_empty(predictions_info), key=lambda x: x["score"], reverse=True)[:num_max_prediction]
        for id, predictions_info in prediction_cadidates_info.items()
    }

    for predictions_info in predictions_info_per_id.values():
        scores = np.array([prediction_info.pop("score") for prediction_info in predictions_info])
        probabilities = scores / scores.sum()
        for probability, prediction_info in zip(probabilities, predictions_info):
            prediction_info["probability"] = probability

    with open(path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as json_output:
        best_predictions = {
            id: prediction_info[0]["text"] for id, prediction_info in predictions_info_per_id.items()
        }
        json.dump(best_predictions, json_output, ensure_ascii=False, indent=4)
    with open(
        path.join(args.output_dir, f"top_{num_max_prediction}_predictions.json"), "w", encoding="utf-8"
    ) as json_output:
        top_n_predictions = {
            id: [
                {
                    k: float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v
                    for k, v in info.items()
                }
                for info in prediction_info
            ]
            for id, prediction_info in predictions_info_per_id.items()
        }
        json.dump(top_n_predictions, json_output, ensure_ascii=False, indent=4)

    predictions = [
        {"id": id, "prediction_text": predictions_info[0]["text"]}
        for id, predictions_info in predictions_info_per_id.items()
    ]
    references = [
        {"id": example["id"], "answers": example["answers"]} for example in args.dataset["validation"]
    ]
    print(predictions)
    print(references)
    metric = load_metric("squad")
    return metric.compute(predictions=predictions, references=references)


def compute_metrics_g(args, eval_predictions):
    dataset = args.dataset["validation"]
    preds, labels = eval_predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    for i in range(len(preds)):
        preds[i] = list(map(lambda x : 0 if x == 250099 else x, preds[i]))
    for i in range(len(labels)):
        labels[i] = list(map(lambda x : 0 if x == -100 else x, labels[i]))

    decoded_preds = args.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = args.tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 간단한 post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    prediction_cadidates_info = defaultdict(list)
    for pred, ans, overflow_to_sample_mapping in zip(decoded_preds, decoded_labels, args.processed_eval_dataset["overflow_to_sample_mapping"]):
        id = dataset["id"][overflow_to_sample_mapping]
        context = dataset["context"][overflow_to_sample_mapping]
        question = dataset["question"][overflow_to_sample_mapping]
        prediction_cadidates_info[id].append(
                        {
                            "text": context,
                            "question": question,
                            "prediction": pred,
                            "answer": ans
                        }
                    )
    """
    predictions_info_per_id = {
        id: predictions_info
        for id, predictions_info in prediction_cadidates_info.items()
    }
    """
    formatted_predictions = []
    references = []

    for id in prediction_cadidates_info.keys():
        formatted_predictions.append({"id": id, "prediction_text": prediction_cadidates_info[id][0]["prediction"]})
        references.append({"id": id, "answers": {"text": prediction_cadidates_info[id][0]["answer"], "answer_start" : [0]}})
    
    metric = load_metric("squad")

    with open(path.join(args.output_dir, "generation.json"), "w", encoding="utf-8") as json_output:
        json.dump(prediction_cadidates_info, json_output, ensure_ascii=False, indent=4)
    
    return metric.compute(predictions=formatted_predictions, references=references)

def postprocess_text(preds, labels):
    
    preds = [re.sub(r"(\\n|\\|\n|\'|\.|\"|<extra_id_[0-9]>|\?)", "", pred) for pred in preds]
    preds = [re.sub(r"\s\s+", " ", pred) for pred in preds]
    labels = [re.sub(r"(\'|\.|\")", "", label) for label in labels]
    
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels