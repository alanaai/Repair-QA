import torch
import timeit
import os
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from app.domain.model.get_tokenizer import get_tokenizer
from app.trainer.evaluate import evaluate

def train_model(train_iter, dev_iter, tokenizer, model, config):

    model.train()
    if torch.cuda.is_available():
        model.cuda()

    num_epoch = config.MAX_EPOCH
    t_total = len(train_iter) // config.GRAD_STEPS * num_epoch
    warmup_proportion = 0.1
    num_warmup_steps = int(t_total * warmup_proportion)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE, eps=config.EPS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    optimizer.zero_grad()
    model.zero_grad()
    training_loss = []

    start_time = timeit.default_timer()

    for epoch in range(num_epoch):
        epoch_iterator = tqdm(train_iter, desc="Iteration", disable=0)
        tr_loss = 0.0

        model.train()
        for step, batch in enumerate(epoch_iterator):
            src_ids, src_mask, _, target = batch
            tgt_ids, _, _ = target
            tgt_ids[tgt_ids == tokenizer.pad_token_id] = -100
            inputs = {'input_ids' : src_ids.to(config.DEVICE),
                    'attention_mask' : src_mask.to(config.DEVICE),
                    'labels': tgt_ids.to(config.DEVICE)
                    }

            output= model(**inputs)
            loss = output['loss']

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % config.GRAD_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            epoch_iterator.set_description("Epoch: {}/{}, Loss: {:.5f}".format(epoch+1, num_epoch, tr_loss/(step + 1)))

        training_loss.append(tr_loss/len(train_iter))

        savefile = os.path.join(config.PRED_SAVE_DIR, 'prediction_'+str(epoch)+'.json')

        evaluate(model, dev_iter, tokenizer, config, savefile=savefile)

    torch.save(model, os.path.join(config.MODEL_SAVE_DIR, 'models'))

    total_time = timeit.default_timer() - start_time
    print('Total training time: ',   total_time)
    return training_loss

def load_model(config):

    tokenizer = get_tokenizer(config.MODEL)

    # load trained tokenizer and model
    tokenizer = tokenizer.from_pretrained(os.path.join(config.MODEL_SAVE_DIR, 'tokenizer'))

    model = torch.load(
        os.path.join(config.MODEL_SAVE_DIR, 'models'),
        map_location=None if torch.cuda.is_available() else torch.device('cpu'),
    )

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    return tokenizer, model