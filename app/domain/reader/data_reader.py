import torch
import json
import numpy as np
from torch.utils import data

def corpus_reader(path):
    context, utterance, target = [], [], []
    data = json.load(open(path, 'r'))
    for d in data:
        ctx = [d['question'], d['long-answer']]
        context.append(ctx)
        utterance.append(d['TPR-1'])
        if 'Ref-Rewrite' in d:
            target.append(d['Ref-Rewrite'])
    if len(target) == 0:
        target = None
    return context, utterance, target

class qr_Dataset(data.Dataset):
    def __init__(self, context, utterance, target, tokenizer, do_lower_case=False):
        self.context = context
        self.utterance = utterance
        self.target = target
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.lower = do_lower_case

    def __len__(self):
        return len(self.utterance)

    def process_input(self, text):
        text = text.strip()
        if self.lower:
            return text.lower()
        else:
            return text

    def __getitem__(self, idx):
        source = 'summarize: '
        prefix = ['<USR>', '<SYS>']
        if len(self.context[idx])%2:
            prefix = prefix[::-1]
        for i, turn in enumerate(self.context[idx]):
            source += prefix[i%2] + ' ' + self.process_input(turn) + ' '

        source += '<CUR>' + ' ' + self.process_input(self.utterance[idx]) + self.tokenizer.eos_token
        source = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(source)[-256:])

        if self.target is not None:
            target = self.process_input(self.target[idx]) + ' ' + self.tokenizer.eos_token
            target = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(target))

            return source, len(source), self.process_input(self.utterance[idx]), target, len(target), self.process_input(self.target[idx])
        else:
            return source, len(source), self.process_input(self.utterance[idx]), None


def pad(batch):
    get_element = lambda x: [sample[x] for sample in batch]

    # pad function
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]

    # input utterance
    src_token = get_element(2)
    # get input sequence length
    src_len = get_element(1)
    src_maxlen = np.array(src_len).max()
    # do pad for src
    src_ids = do_pad(0, src_maxlen)
    # attention mask for input
    src_mask = [[(i>0) for i in ids] for ids in src_ids]

    LT = torch.LongTensor
    src_ids_len = LT(list(map(len, src_ids)))
    _, sorted_idx = src_ids_len.sort(0, descending=True)

    src_ids = LT(src_ids)[sorted_idx]
    src_mask = LT(src_mask)[sorted_idx]

    if get_element(3)[0] is not None:
        tgt_token = get_element(5)
        tgt_len = get_element(4)
        tgt_maxlen = np.array(tgt_len).max()
        tgt_ids = do_pad(3, tgt_maxlen)
        tgt_mask = [[(i>0) for i in ids] for ids in tgt_ids]

        tgt_ids = LT(tgt_ids)[sorted_idx]
        tgt_mask = LT(tgt_mask)[sorted_idx]

        return src_ids, src_mask, src_token, (tgt_ids, tgt_mask, tgt_token)

    else:
        return src_ids, src_mask, src_token, None

def get_data_iterator(config, tokenizer, datapath, shuffle=True):
    ctx, utt, tgt = corpus_reader(datapath)
    dataset = qr_Dataset(ctx, utt, tgt, tokenizer, do_lower_case=config.IS_LOWER)

    data_iter = data.DataLoader(dataset=dataset,
                                batch_size=config.BATCH_SIZE,
                                shuffle=shuffle,
                                num_workers=2,
                                collate_fn=pad)

    return data_iter

def generate_training_data(config, tokenizer):
    training_datapath, validation_datapath = config.TRAINING_DATA, config.DEV_DATA

    train_iter = get_data_iterator(config, tokenizer, training_datapath)

    dev_iter = get_data_iterator(config, tokenizer, validation_datapath, shuffle=False)

    return train_iter, dev_iter
