
def get_tokenizer(model_name):
    if 't5' in model_name:
        from app.domain.model.t5 import TOKENIZER
    else:
        print('import model error')
        exit()

    return TOKENIZER