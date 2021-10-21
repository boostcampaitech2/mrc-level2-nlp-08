from bisect import bisect_left


def preprocess(args, examples):
    answers = examples["answers"]
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    examples["start_positions"] = []
    examples["end_positions"] = []
    for i, (input_ids, token_type_ids, offset_mapping, overflow_to_sample_mapping) in enumerate(
        zip(
            examples["input_ids"],
            examples["token_type_ids"],
            examples["offset_mapping"],
            examples["overflow_to_sample_mapping"],
        )
    ):
        cls_token_idx = input_ids.index(args.tokenizer.cls_token_id)
        answer_token_start_idx = answer_token_end_idx = cls_token_idx

        token_type_ids = examples.sequence_ids(i)
        examples["token_type_ids"][i] = token_type_ids

        answer_info = answers[overflow_to_sample_mapping]
        if answer_info:
            answer_start_idx = answer_info["answer_start"][0]
            answer_end_idx = answer_start_idx + len(answer_info["text"][0])

            context_token_start_idx = token_type_ids.index(1)
            # Additional step forward(last index - 1) to exclude the last special token
            context_token_end_idx = len(token_type_ids) - 2

            offset_start_idxs, offset_end_idxs = zip(*offset_mapping)
            if (
                answer_start_idx >= offset_start_idxs[context_token_start_idx]
                and answer_end_idx <= offset_end_idxs[context_token_end_idx]
            ):
                answer_token_start_idx = context_token_start_idx + bisect_left(
                    offset_start_idxs[context_token_start_idx:context_token_end_idx], answer_start_idx
                )
                answer_token_end_idx = context_token_start_idx + bisect_left(
                    offset_end_idxs[context_token_start_idx:context_token_end_idx], answer_end_idx
                )

        examples["start_positions"].append(answer_token_start_idx)
        examples["end_positions"].append(answer_token_end_idx)

    args.token_type_ids = examples["token_type_ids"]
    if "roberta" in args.config.model_type.lower():
        examples.pop("token_type_ids")
    return examples


def preprocess_testset(args, examples):
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    args.token_type_ids = examples["token_type_ids"]
    if "roberta" in args.config.model_type.lower():
        examples.pop("token_type_ids")
    return examples

def preprocess_g(args, examples):
    # Prepocessing Query Part
    formatted_question = list(map(lambda text: f"question: {text} context:", examples["question"]))
    model_inputs = args.tokenizer(
        formatted_question,
        examples["context"],
        truncation="only_second",
        max_length=args.max_length-1,
        stride=args.stride,
        return_overflowing_tokens=True,
        add_special_tokens=False
    )
    # add </s> at the end of the sentence
    model_inputs["input_ids"] = list(map(lambda id: id + [1], model_inputs["input_ids"]))
    # attention mask
    model_inputs["attention_mask"] = list(map(lambda mask: mask + [1], model_inputs["attention_mask"]))

    # Preprocessing Answers
    answers = [f'{a["text"][0]} </s>' for a in examples['answers']]
    
    with args.tokenizer.as_target_tokenizer():
        targets = args.tokenizer(answers, add_special_tokens=False)


    model_inputs["labels"] = []
    model_inputs["id"] = []
    for i in model_inputs["overflow_to_sample_mapping"]:
        model_inputs["labels"].append(targets["input_ids"][i])
        model_inputs["id"].append(examples["id"][i])
    return model_inputs