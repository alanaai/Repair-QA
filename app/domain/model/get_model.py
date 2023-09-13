
def get_model(model_name):
    if 't5' in model_name:
        from app.domain.model.t5 import MODEL
    else:
        print('import model error')
        exit()

    return MODEL