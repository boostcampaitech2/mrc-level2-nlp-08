from collections import defaultdict

from datasets import load_metric
from transformers.trainer_utils import EvalPrediction

import numpy as np


def check_empty(prediction: list):
    if prediction:
        return prediction
    return [{"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}]


def compute_metrics(args, outputs: EvalPrediction):
    logits_of_start_idxs_predictions, logits_of_end_idxs_predictions = outputs.predictions
    n_best_predictions = 20

    dataset = args.dataset["validation"]
    prediction_cadidates_info = defaultdict(list)
    for (
        overflow_to_sample_mapping,
        token_type_ids,
        offset_mapping,
        logits_of_start_idxs_prediction,
        logits_of_end_idxs_prediction,
    ) in zip(
        args.processed_eval_dataset["overflow_to_sample_mapping"],
        args.processed_eval_dataset["token_type_ids"],
        args.processed_eval_dataset["offset_mapping"],
        logits_of_start_idxs_predictions,
        logits_of_end_idxs_predictions,
    ):
        start_offset_idxs = np.argsort(logits_of_start_idxs_prediction)[-n_best_predictions:][::-1].tolist()
        end_offset_idxs = np.argsort(logits_of_end_idxs_prediction)[-n_best_predictions:][::-1].tolist()
        for start_offset_idx in start_offset_idxs:
            for end_offset_idx in end_offset_idxs:
                if (
                    start_offset_idx < len(offset_mapping)
                    and start_offset_idx <= end_offset_idx < len(offset_mapping)
                    and token_type_ids[start_offset_idx] == 1
                    and token_type_ids[end_offset_idx] == 1
                    and 0
                    # <= offset_mapping[end_offset_idx][1] - offset_mapping[start_offset_idx][0]    # by original sequence length
                    <= end_offset_idx - start_offset_idx                                            # by number of tokens
                    <= args.max_answer_length
                ):
                    prediction_cadidates_info[dataset["id"][overflow_to_sample_mapping]].append(
                        {
                            "text": dataset["context"][overflow_to_sample_mapping][
                                offset_mapping[start_offset_idx][0] : offset_mapping[end_offset_idx][1]
                            ],
                            "start_logit": logits_of_start_idxs_prediction[start_offset_idx],
                            "end_logit": logits_of_end_idxs_prediction[end_offset_idx],
                            "score": logits_of_start_idxs_prediction[start_offset_idx]
                            + logits_of_end_idxs_prediction[end_offset_idx],
                        }
                    )

    prediction_info_per_id = {
        id: sorted(check_empty(predictions_info), key=lambda x: x["score"], reverse=True)[:n_best_predictions]
        for id, predictions_info in prediction_cadidates_info.items()
    }

    for predictions_info in prediction_info_per_id.values():
        scores = np.array([prediction_info.pop("score") for prediction_info in predictions_info])
        probabilities = scores / scores.sum()
        for probability, prediction_info in zip(probabilities, predictions_info):
            prediction_info["probability"] = probability

    predictions = [
        {"id": id, "prediction_text": predictions_info[0]["text"]}
        for id, predictions_info in prediction_info_per_id.items()
    ]
    references = [
        {"id": example["id"], "answers": example["answers"]} for example in args.dataset["validation"]
    ]
    metric = load_metric("squad")
    return metric.compute(predictions=predictions, references=references)
