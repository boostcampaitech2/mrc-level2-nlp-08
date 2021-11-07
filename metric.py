from collections import defaultdict
from os import path
import json

from datasets import load_metric

import numpy as np
import re

def compute_metrics_g(args, eval_predictions):
    predictions = postprocess_g(args, eval_predictions)
    references = [{"id": example["id"], "answers": example["answers"]} for example in args.dataset["validation"]]
    metric = load_metric("squad")

    return metric.compute(predictions=predictions, references=references)

def postprocess_g(args, eval_predictions):
    dataset = args.dataset["validation"]
    preds, labels, scores = eval_predictions
    
    labels = np.where(labels != -100, labels, args.tokenizer.pad_token_id)

    decoded_preds = args.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = args.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    prediction_cadidates_info = defaultdict(list)

    for pred, ans, score, overflow_to_sample_mapping in zip(decoded_preds, decoded_labels, scores, args.processed_eval_dataset["overflow_to_sample_mapping"]):
        id = dataset["id"][overflow_to_sample_mapping]
        question = dataset["question"][overflow_to_sample_mapping]
        if ans == "":
            ans = "Answer Not Found"
        prediction_cadidates_info[id].append(
                        {
                            "question": question,
                            "prediction": pred,
                            "answer": ans,
                            "score": float(score) if isinstance(score, (np.float16, np.float32, np.float64)) else score
                        }
                    )
    
    formatted_predictions = []
    
    for id in prediction_cadidates_info.keys():
        sorted_pred = sorted(prediction_cadidates_info[id], key=lambda x: x["score"], reverse=True)
        text_found = False
        for pred in sorted_pred:
            if pred["prediction"] != "Answer Not Found":
                final_text = pred["prediction"]
                text_found = True
                break
        formatted_predictions.append({"id": id, "prediction_text": final_text if text_found else sorted_pred[0]["prediction"]})
    
    ## Get the best prediction out of candidates ##
    ## for the information, look generation json, and look prediction json for final submission ##

    prediction_json = {}
    for prediction in formatted_predictions:
        final_prediction = prediction["prediction_text"]
        prediction_cadidates_info[prediction["id"]] = [f"final prediction : {final_prediction}"] + prediction_cadidates_info[prediction["id"]]
        prediction_json[prediction["id"]] = prediction["prediction_text"]

    with open(path.join(args.output_dir, "generation.json"), "w", encoding="utf-8") as json_output:
        json.dump(prediction_cadidates_info, json_output, ensure_ascii=False, indent=4)

    with open(path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as json_output:
        json.dump(prediction_json, json_output, ensure_ascii=False, indent=4)

    return formatted_predictions


def postprocess_text(preds, labels):
    preds = [re.sub(r"(\\n|\\|\n|\'|\.|\"|<extra_id_[0-9]>|\?)", "", pred) for pred in preds]
    preds = [re.sub(r"\s\s+", " ", pred) for pred in preds]
    labels = [re.sub(r"(\'|\.|\")", "", label) for label in labels]
    
    preds = [pred.strip() for pred in preds]
    preds = ["Answer Not Found" if pred == "" else pred for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def postprocess_g_test(args, predictions):
    dataset = args.dataset["validation"]
    preds, scores = predictions
    
    decoded_preds = args.tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    decoded_preds = postprocess_test_text(decoded_preds)
    prediction_cadidates_info = defaultdict(list)

    for pred, score, overflow_to_sample_mapping in zip(decoded_preds, scores, args.processed_eval_dataset["overflow_to_sample_mapping"]):
        id = dataset["id"][overflow_to_sample_mapping]
        question = dataset["question"][overflow_to_sample_mapping]
        prediction_cadidates_info[id].append(
                        {
                            "question": question,
                            "prediction": pred,
                            "score": float(score) if isinstance(score, (np.float16, np.float32, np.float64)) else score
                        }
                    )
    
    formatted_predictions = []
    
    for id in prediction_cadidates_info.keys():
        sorted_pred = sorted(prediction_cadidates_info[id], key=lambda x: x["score"], reverse=True)
        text_found = False
        for pred in sorted_pred:
            if pred["prediction"] != "Answer Not Found":
                final_text = pred["prediction"]
                text_found = True
                break
        formatted_predictions.append({"id": id, "prediction_text": final_text if text_found else sorted_pred[0]["prediction"]})
    
    prediction_json = {}
    for prediction in formatted_predictions:
        final_prediction = prediction["prediction_text"]
        prediction_cadidates_info[prediction["id"]] = [f"final prediction : {final_prediction}"] + prediction_cadidates_info[prediction["id"]]
        prediction_json[prediction["id"]] = prediction["prediction_text"]

    with open(path.join(args.output_dir, "generation.json"), "w", encoding="utf-8") as json_output:
        json.dump(prediction_cadidates_info, json_output, ensure_ascii=False, indent=4)

    with open(path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as json_output:
        json.dump(prediction_json, json_output, ensure_ascii=False, indent=4)

    return formatted_predictions


def postprocess_test_text(preds):
    preds = [re.sub(r"(\\n|\\|\n|\'|\.|\"|<extra_id_[0-9]>|\?)", "", pred) for pred in preds]
    preds = [re.sub(r"\s\s+", " ", pred) for pred in preds]
    preds = [pred.strip() for pred in preds]
    preds = ["Answer Not Found" if pred == "" else pred for pred in preds]
    return preds