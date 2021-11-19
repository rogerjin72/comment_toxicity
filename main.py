import os
import torch
import pandas as pd
import argparse

from models.bert_ranking import BertForRanking
from models.dataset import RankingDataset
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments
)

def load_model(finetuned=None):
    base_model = args.from_pretrained
    tokenizer = BertTokenizer.from_pretrained(base_model, model_max_length=args.sequence_length)
    if not finetuned:
        model = BertForRanking.from_pretrained(base_model)
    else:
        model = BertForRanking.from_pretrained(finetuned)
    return tokenizer, model


def train():
    tokenizer, model = load_model()

    try:
        train_set = torch.load('cache/dataset.pt')

    except FileNotFoundError:
        train_set = pd.read_csv('processed_data/train.csv')
        first = tokenizer(train_set['s1'].to_list(), return_tensors='pt', truncation=True, padding=True)
        second = tokenizer(train_set['s2'].to_list(), return_tensors='pt', truncation=True, padding=True)
        values = train_set['beta'].to_numpy()
        train_set = RankingDataset(first, second, values)
        if not os.path.isdir('cache'):
            os.mkdir('cache')

        torch.save(train_set, 'cache/dataset.pt')

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=0.1,
        save_strategy='epoch' if args.save_model else 'no',
        logging_steps=25
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set
    )

    trainer.train()


def predict():
    tokenizer, model = load_model(args.prediction_model)
    model = model.to(device)

    test = pd.read_csv('data/comments_to_score.csv')
    test_set = RankingDataset(test['text'])

    for i in range(len(test)):
        print(model(**test_set[i]))
        break


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train',
        type=int,
        default=1
    )

    parser.add_argument(
        '--predict',
        type=int,
        default=1
    )

    parser.add_argument(
        '--prediction_model',
        type=str,
        default='trained_model/checkpoint-474'
    )

    parser.add_argument(
        '--from_pretrained',
        type=str,
        help='Path to directory containing checkpoint',
        default='bert-base-cased'
    )

    parser.add_argument(
        '--sequence_length',
        type=int,
        default=256
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Training batch size per GPU',
        default=32
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Optimizer learning rate',
        default=5e-5
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help="Number of epochs to train for",
        default=3
    )

    parser.add_argument(
        '--save_model',
        type=int,
        help='save checkpoints of the model',
        default=0
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='directory to save the model',
        default='trained_model'
    )

    return parser


parser = arg_parser()
args, _ = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if args.train:
        train()
    if args.predict:
        predict()