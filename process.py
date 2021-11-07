def preprocess_g(args, examples):
    formatted_question = list(map(lambda text: f"[질문]: {text} [지문]: ", examples["question"]))
    model_inputs = args.tokenizer(
        formatted_question,
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    answers = [f'{a["text"][0]} </s>' for a in examples['answers']]
    with args.tokenizer.as_target_tokenizer():
        targets = args.tokenizer(answers)
        nonestring = args.tokenizer('</s>')

    model_inputs["labels"] = []
    for index, sample_mapping in enumerate(model_inputs["overflow_to_sample_mapping"]):
        answer_start, answer_end = examples['answers'][sample_mapping]['answer_start'][0], examples['answers'][sample_mapping]['answer_start'][0] + len(examples['answers'][sample_mapping]['text'][0])
        context_start, context_end = model_inputs["offset_mapping"][index][0][0], model_inputs["offset_mapping"][index][-1][1]
        if answer_start >= context_start and answer_end <= context_end:
            model_inputs["labels"].append(targets["input_ids"][sample_mapping])
        else:
            model_inputs["labels"].append(nonestring['input_ids'])
    
    return model_inputs


def preprocess_testset_g(args, examples):
    formatted_question = list(map(lambda text: f"[질문]: {text} [지문]: ", examples["question"]))
    examples = args.tokenizer(
        formatted_question,
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
    )

    return examples