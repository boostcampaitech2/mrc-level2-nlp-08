def preprocess_e(args, examples):
    # Prepocessing Query Part
    formatted_question = list(map(lambda text: f"[질문]: {text} [지문]: ", examples["question"]))
    model_inputs = args.tokenizer(
        formatted_question,
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        padding=True,
    )

    # Preprocessing Answers
    # For each shards of context, I'll put None'</s>' if the answer doesn't exist in the context.
    answers = [f'{a["text"][0]}' for a in examples['answers']]
    with args.tokenizer.as_target_tokenizer():
        targets = args.tokenizer(answers, padding='max_length', max_length=50)
        nonestring = args.tokenizer('', padding='max_length', max_length=50)

    model_inputs["labels"] = []
    for index, sample_mapping in enumerate(model_inputs["overflow_to_sample_mapping"]):
        if args.tokenizer.decode(model_inputs["input_ids"][index]).find(examples['answers'][sample_mapping]['text'][0]) != -1:
            model_inputs["labels"].append(targets["input_ids"][sample_mapping])
        else:
            model_inputs["labels"].append(nonestring['input_ids'])
    model_inputs["decoder_input_ids"] = model_inputs["labels"]

    return model_inputs