import os
import argparse

from app.config import config
from app.domain.reader import data_reader
from app.domain.model.get_model import get_model
from app.domain.model.get_tokenizer import get_tokenizer
from app.trainer.train import train_model

config.set_seed(config.SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disfluency Tagger')
    parser.add_argument('--seed', type=int, dest='SEED', required=False)
    parser.add_argument('--data_dir', type=str, dest='DATA_DIR', required=False)
    parser.add_argument('--model_name', type=str, dest='MODEL', required=False)
    parser.add_argument('--model_dir', type=str, dest='MODEL_SAVE_DIR', required=False)
    parser.add_argument('--pred_dir', type=str, dest='PRED_SAVE_DIR', required=False)
    parser.add_argument('--lr', type=float, dest='LEARNING_RATE', required=False)
    parser.add_argument('--bz', type=int, dest='BATCH_SIZE', required=False)
    parser.add_argument('--epoch', type=int, dest='MAX_EPOCH', required=False)
    parser.add_argument('--earlystop', type=int, dest='EARLY_STOP', required=False)
    parser.add_argument('--gradclip', type=float, dest='GRAD_CLIP', required=False)
    args = parser.parse_args()
    for k in vars(args):
        if getattr(args, k):
            setattr(config, k, getattr(args, k))
    print('Model: {}'.format(config.MODEL))

    print("Loading Model....")
    tokenizer = get_tokenizer(config.MODEL)
    model = get_model(config.MODEL)

    tokenizer = tokenizer.from_pretrained(config.MODEL)
    model = model.from_pretrained(config.MODEL)

    # add special tokens to tokenizer
    special_tokens_dict = {"sep_token": "[SEP]", "eos_token": "</s>"}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f'Added {num_added_toks} new tokens in vocab.')

    tokenizer.save_pretrained(os.path.join(config.MODEL_SAVE_DIR, "tokenizer"))

    model.resize_token_embeddings(len(tokenizer))
    print("Loading data....")
    train_iter, dev_iter = data_reader.generate_training_data(config, tokenizer)

    print(f'Using Device: {config.DEVICE}')

    print(f'Starting training of {config.MODEL}...')
    train_model(train_iter, dev_iter, tokenizer, model, config)
