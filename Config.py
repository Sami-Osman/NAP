import torch
import torch.nn as nn
import numpy as np
# this file is used to set the basic parameters of the model under the training or prediction process.
class Config:
    bert_model = 'albert-base-v2' # 'albert-base-v2'  #'bert-base-uncased'  #"distilbert-base-uncased"
    bert_version = 2 # [0 is baseModel, 1 is with base+LSTM , 2 is base+GRU]
    freeze_bert = True  # if True, freeze the encoder weights and only update the classification layer weights
    maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = 16  # batch size
    lr = 2e-5  # learning rate
    epochs = 5  # number of training epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    with_labels = True #False if the dataset is without labels 0 and 1

