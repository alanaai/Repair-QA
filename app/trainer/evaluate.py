import os
import json
import torch
from tqdm import tqdm

def evaluate(model, dev_iter, tokenizer, config, savefile='pred.json'):
    model.eval()

    with torch.no_grad():
        data_iterator = tqdm(dev_iter, desc="Predicting", disable=0)

        to_write = []
        for batch in data_iterator:
            src_ids, src_mask, src_token, target = batch

            ids = src_ids.to(config.DEVICE)
            mask = src_mask.to(config.DEVICE)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=50,
                num_beams=config.BEAM_SIZE,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

            tgt_token = None
            if target is not None:
                _, _, tgt_token = target

            if tgt_token is not None:
                for utt, hyp, ref in zip(src_token, preds, tgt_token):
                    to_write.append({"input": utt,
                                     "hypothesis" : hyp,
                                     "reference": ref})
            else:
                for utt, hyp in zip(src_token, preds):
                    to_write.append({"input": utt,
                                     "hypothesis" : hyp})

    with open(savefile, 'w') as fout:
        json.dump(to_write, fout, indent=4)

    model.train()