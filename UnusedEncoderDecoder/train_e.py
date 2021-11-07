def train_e(settings, args):
    args.encoder_config = AutoConfig.from_pretrained("klue/bert-base")
    args.decoder_config = AutoConfig.from_pretrained("klue/bert-base", is_decoder=True, add_cross_attention=True, decoder_start_token_id=2)
    args.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", padding=True)
    args.config = EncoderDecoderConfig.from_encoder_decoder_configs(args.encoder_config, args.decoder_config)
    model = EncoderDecoderModel(args.config)

    args.dataset = load_from_disk(settings.trainset_path)
    
    train_dataset = args.dataset["train"]
    column_names = train_dataset.column_names
    train_dataset = train_dataset.map(
        send_along(preprocess_e, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )

    eval_dataset = args.dataset["validation"]
    eval_dataset = eval_dataset.map(
        send_along(preprocess_e, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )
    args.processed_eval_dataset = eval_dataset
    trainer = MySeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=args.tokenizer,
        compute_metrics=send_along(compute_metrics_g, sent_along=args),
    )
    trainer.train()
    trainer.save_model()
    trainer.evaluate(
        max_length=args.max_answer_length,
        num_beams=args.num_beams,
        metric_key_prefix="eval"
    )