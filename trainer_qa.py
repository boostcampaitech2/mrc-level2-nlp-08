# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""

from typing import Dict, List, Optional
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import (
    PredictionOutput,
    speed_metrics,
    EvalPrediction,
)
import time
import math
from torch.utils.data import DataLoader, Dataset

if is_datasets_available():
    import datasets


# NOTE
from utils_qa import postprocess_qa_predictions, check_no_error


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, data_args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.data_args = data_args

    def post_processing(self, predictions, phase="train"):
        predictions = postprocess_qa_predictions(
            examples=self.eval_examples,
            features=self.eval_dataset,
            predictions=predictions,
            max_answer_length=self.data_args.max_answer_length,
            output_dir=self.args.output_dir,
            prefix=None if phase == "train" else str(self.data_args.top_k_retrieval),
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if phase == "test":
            return formatted_predictions
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.eval_examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.post_processing is not None and self.compute_metrics is not None:
            eval_preds = self.post_processing(output.predictions)
            squad_metrics = self.compute_metrics(eval_preds)
            squad_metrics = {"eval_" + k: v for k, v in squad_metrics.items()}
            output.metrics.update(squad_metrics)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:

        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        predictions = self.post_processing(output.predictions, phase="test")
        return predictions
