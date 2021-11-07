from bisect import bisect_left

def special_token_adder()


def preprocess(args, examples):
    answers = examples["answers"]
    examples["question"] = ['<s> ' + q + '</s>'for q in examples['question']]
    #examples["context"] = [ ' </s> ' + c + ' </s>' for c in examples['context']]
    examples = args.tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    print(examples['input_ids'])
    examples["start_positions"] = []
    examples["end_positions"] = []
    for i, (input_ids, offset_mapping, overflow_to_sample_mapping) in enumerate(
        zip(
            examples["input_ids"],
            #examples["token_type_ids"],#
            examples["offset_mapping"],
            examples["overflow_to_sample_mapping"],
        )
    ):
        #print(input_ids, args.tokenizer.cls_token_id)
        cls_token_idx = input_ids.index(args.tokenizer.cls_token_id)
        answer_token_start_idx = answer_token_end_idx = cls_token_idx

        token_type_ids = examples.sequence_ids(i)
        #examples["token_type_ids"][i] = token_type_ids #

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

    #args.token_type_ids = examples["token_type_ids"]#
    #if "roberta" in args.config.model_type.lower():#
    #    examples.pop("token_type_ids")
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
