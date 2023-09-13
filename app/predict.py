import sys
from app.config import config
from app.trainer.train import load_model
from app.domain.reader import data_reader
from app.trainer.evaluate import evaluate


if __name__ == '__main__':
    data_file = sys.argv[1]
    savefile = sys.argv[2]

    print("Loading Model....")
    tokenizer, model = load_model(config)

    data_iter = data_reader.get_data_iterator(config, tokenizer, data_file, shuffle=False)

    evaluate(model, data_iter, tokenizer, config, savefile=savefile)