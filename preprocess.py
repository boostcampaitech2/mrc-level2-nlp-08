from utils import check_and_get_max_sequence_length


def make_datasets(datasets, tokenizer, training_args, data_args):
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")

    train_dataset = preprocess_train_datasets(datasets, tokenizer, data_args)
    eval_dataset = preprocess_eval_datasets(datasets, tokenizer, data_args)

    return train_dataset, eval_dataset


def preprocess_train_datasets(datasets, tokenizer, data_args):
    train_column_names = datasets["train"].column_names

    train_question_column_name = "question" if "question" in train_column_names else train_column_names[0]
    train_context_column_name = "context" if "context" in train_column_names else train_column_names[1]
    train_answer_column_name = "answers" if "answers" in train_column_names else train_column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = check_and_get_max_sequence_length(data_args, tokenizer)

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[train_question_column_name if pad_on_right else train_context_column_name],
            examples[train_context_column_name if pad_on_right else train_question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            sequence_ids = tokenized_examples.sequence_ids(i)  # token_type_ids

            sample_index = sample_mapping[i]
            answers = examples[train_answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    train_dataset = datasets["train"]

    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    return train_dataset


def preprocess_eval_datasets(datasets, tokenizer, data_args):
    eval_column_names = datasets["validation"].column_names

    eval_question_column_name = "question" if "question" in eval_column_names else eval_column_names[0]
    eval_context_column_name = "context" if "context" in eval_column_names else eval_column_names[1]
    eval_answer_column_name = "answers" if "answers" in eval_column_names else eval_column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = check_and_get_max_sequence_length(data_args, tokenizer)

    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[eval_question_column_name if pad_on_right else eval_context_column_name],
            examples[eval_context_column_name if pad_on_right else eval_question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    return eval_dataset
