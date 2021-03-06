from bisect import bisect_left


def preprocess(args, examples):
    answers = examples["answers"]
    # print(examples["question"][0])
    # print(examples["context"][0])
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        # padding="max_length",  #
    )
    # print(examples["token_type_ids"][0])

    examples["start_positions"] = []
    examples["end_positions"] = []
    for i, (
        input_ids,
        token_type_ids,
        offset_mapping,
        overflow_to_sample_mapping,
    ) in enumerate(
        zip(
            examples["input_ids"],
            examples["token_type_ids"],
            examples["offset_mapping"],
            examples["overflow_to_sample_mapping"],
        )
    ):
        cls_token_idx = input_ids.index(args.tokenizer.cls_token_id)
        answer_token_start_idx = answer_token_end_idx = cls_token_idx
        # print(examples["token_type_ids"][i])
        # print(len(examples.sequence_ids(i)))
        # token_type_ids = examples.sequence_ids(i)
        token_type_ids = examples.sequence_ids(i)
        examples["token_type_ids"][i] = token_type_ids
        # print(token_type_ids)
        # print(token_type_ids.index(1))
        # print(examples.sequence_ids(i).index(1))

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
                    offset_start_idxs[context_token_start_idx:context_token_end_idx],
                    answer_start_idx,
                )
                answer_token_end_idx = context_token_start_idx + bisect_left(
                    offset_end_idxs[context_token_start_idx:context_token_end_idx],
                    answer_end_idx,
                )

        examples["start_positions"].append(answer_token_start_idx)
        examples["end_positions"].append(answer_token_end_idx)
    # print(examples["start_positions"])
    # print(examples["end_positions"])
    args.token_type_ids = examples["token_type_ids"]
    # print(examples["token_type_ids"])
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


def preprocess_temp(args, examples):
    answers = examples["answers"]
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        # padding="max_length",  #
    )
    # print(examples)
    sample_mapping = examples.pop("overflow_to_sample_mapping")
    offset_mapping = examples.pop("offset_mapping")

    examples["start_positions"] = []
    examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = examples["input_ids"][i]
        cls_index = input_ids.index(args.tokenizer.cls_token_id)  # cls index

        # sequence id??? ??????????????? (to know what is the context and what is the question).
        sequence_ids = examples.sequence_ids(i)

        # ????????? example??? ???????????? span??? ?????? ??? ????????????.
        sample_index = sample_mapping[i]
        # print(examples)
        answers_info = answers[sample_index]

        # answer??? ?????? ?????? cls_index??? answer??? ???????????????(== example?????? ????????? ?????? ?????? ????????? ??? ??????).
        if len(answers_info["answer_start"]) == 0:
            examples["start_positions"].append(cls_index)
            examples["end_positions"].append(cls_index)
        else:
            # text?????? ????????? Start/end character index
            start_char = answers_info["answer_start"][0]
            end_char = start_char + len(answers_info["text"][0])

            # text?????? current span??? Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # text?????? current span??? End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # ????????? span??? ??????????????? ???????????????(????????? ?????? ?????? CLS index??? label????????????).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                examples["start_positions"].append(cls_index)
                examples["end_positions"].append(cls_index)
            else:
                # token_start_index ??? token_end_index??? answer??? ????????? ???????????????.
                # Note: answer??? ????????? ????????? ?????? last offset??? ????????? ??? ????????????(edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                examples["end_positions"].append(token_end_index + 1)
        args.token_type_ids = examples["token_type_ids"]
        print(examples["start_positions"])
        print(examples["end_positions"])
        if "roberta" in args.config.model_type.lower():
            examples.pop("token_type_ids")

        return examples
