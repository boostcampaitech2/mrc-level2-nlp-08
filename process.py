from bisect import bisect_left

<<<<<<< HEAD
def special_token_adder()


def preprocess(args, examples):
    answers = examples["answers"]
    examples["question"] = ['<s> ' + q + '</s>'for q in examples['question']]
    #examples["context"] = [ ' </s> ' + c + ' </s>' for c in examples['context']]
=======

def preprocess(args, examples):
    answers = examples["answers"]
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
<<<<<<< HEAD
        add_special_tokens=True,
    )
    print(examples['input_ids'])
    examples["start_positions"] = []
    examples["end_positions"] = []
    for i, (input_ids, offset_mapping, overflow_to_sample_mapping) in enumerate(
        zip(
            examples["input_ids"],
            #examples["token_type_ids"],#
=======
        # padding="max_length",  #
    )
    # print(examples)

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
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6
            examples["offset_mapping"],
            examples["overflow_to_sample_mapping"],
        )
    ):
<<<<<<< HEAD
        #print(input_ids, args.tokenizer.cls_token_id)
        cls_token_idx = input_ids.index(args.tokenizer.cls_token_id)
        answer_token_start_idx = answer_token_end_idx = cls_token_idx

        token_type_ids = examples.sequence_ids(i)
        #examples["token_type_ids"][i] = token_type_ids #
=======
        cls_token_idx = input_ids.index(args.tokenizer.cls_token_id)
        answer_token_start_idx = answer_token_end_idx = cls_token_idx
        # print(examples["token_type_ids"][i])
        # print(len(examples.sequence_ids(i)))
        # token_type_ids = examples.sequence_ids(i)
        token_type_ids = examples["token_type_ids"][i]
        examples["token_type_ids"][i] = token_type_ids

        #print(token_type_ids.index(1))
        #print(examples.sequence_ids(i).index(1))
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6

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
<<<<<<< HEAD
                    offset_start_idxs[context_token_start_idx:context_token_end_idx], answer_start_idx
                )
                answer_token_end_idx = context_token_start_idx + bisect_left(
                    offset_end_idxs[context_token_start_idx:context_token_end_idx], answer_end_idx
=======
                    offset_start_idxs[context_token_start_idx:context_token_end_idx],
                    answer_start_idx,
                )
                answer_token_end_idx = context_token_start_idx + bisect_left(
                    offset_end_idxs[context_token_start_idx:context_token_end_idx],
                    answer_end_idx,
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6
                )

        examples["start_positions"].append(answer_token_start_idx)
        examples["end_positions"].append(answer_token_end_idx)
<<<<<<< HEAD

    #args.token_type_ids = examples["token_type_ids"]#
    #if "roberta" in args.config.model_type.lower():#
    #    examples.pop("token_type_ids")
=======
    # print(examples["start_positions"])
    # print(examples["end_positions"])
    args.token_type_ids = examples["token_type_ids"]
    # print(examples["token_type_ids"])
    if "roberta" in args.config.model_type.lower():
        examples.pop("token_type_ids")
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6
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
<<<<<<< HEAD
=======


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

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        # print(examples)
        answers_info = answers[sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers_info["answer_start"]) == 0:
            examples["start_positions"].append(cls_index)
            examples["end_positions"].append(cls_index)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers_info["answer_start"][0]
            end_char = start_char + len(answers_info["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                examples["start_positions"].append(cls_index)
                examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
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
>>>>>>> b5419c4995a4dadc4d97b7466588a4490a9089e6
