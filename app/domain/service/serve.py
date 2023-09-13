import torch
import re
from app.config import config
from app.trainer.train import load_model

class Utterance_Rewriter:
    def __init__(self, config):

        self.device = config.DEVICE
        self.tokenizer, self.model = load_model(config)
        self.model.eval()

    def post_process(self, inp):
        return inp

    def process_input(self, text):
        return text.strip().lower()

    def get_source(self, context, utterance):
        source = 'summarize: '
        prefix = ['<USR>', '<SYS>']
        if len(context)%2:
            prefix = prefix[::-1]
        for i, turn in enumerate(context):
            source += prefix[i%2] + ' ' + self.process_input(turn) + ' '

        source += '<CUR>' + ' ' + self.process_input(utterance) + self.tokenizer.eos_token
        source = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(source)[-256:])

        return source

    def predict(self, context, utterance, ctx_len=5, max_len=50, beam_size=config.BEAM_SIZE):

        source = self.get_source(context[-ctx_len:], utterance)

        pred_ids = self.model.generate(
                        input_ids=torch.LongTensor([source]).to(self.device),
                        max_length=max_len,
                        num_beams=beam_size,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        early_stopping=True
                        )[0]

        preds = self.tokenizer.decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return preds

model = Utterance_Rewriter(config)

def get_model():
    return model