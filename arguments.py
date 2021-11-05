import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--num_neg', default=3, type=int)
    parser.add_argument('--train_data', default='../data/train_dataset/train', type=str)
    parser.add_argument('--val_data', default='../data/train_dataset/validation', type=str)
    parser.add_argument('--test_data', default='../data/test_dataset/validation', type=str)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--train_bs', default=4, type=int)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--early_stop', default=5, type=int)
    parser.add_argument('--adam_epsilon', default=1e-08, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)

    return parser