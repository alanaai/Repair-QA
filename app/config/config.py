import os
import random
import torch
import numpy as np

def set_seed(seed):
    print(f'Random Seed: {seed}')
    #random seed
    random.seed(seed)

    #numpy seed
    np.random.seed(seed)

    #torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 123

# DATA directory
DATA_DIR = './data'
TRAINING_DATA = os.path.join(DATA_DIR, 'train.json')
DEV_DATA = os.path.join(DATA_DIR, 'test.json')

# Model to Load ["t5-base"]
MODEL = 't5-base'

# beam size for model inference
BEAM_SIZE = 2

#lower_cased input
IS_LOWER = True

#location to save model files
MODEL_SAVE_DIR = './resources/'
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

#location to save results
PRED_SAVE_DIR = './resources/'
if not os.path.exists(PRED_SAVE_DIR):
    os.makedirs(PRED_SAVE_DIR)


#Training hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5
EPS = 1e-8
BATCH_SIZE = 16
MAX_EPOCH = 5
EARLY_STOP = 10
GRAD_CLIP = 1
GRAD_STEPS = 1