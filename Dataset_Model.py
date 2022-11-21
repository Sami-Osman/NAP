from torch.utils.data import DataLoader, Dataset
import torch
import shutil
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn

class CustomDataset(Dataset):

  def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

    self.data = data  # pandas dataframe
    #Initialize the tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    self.maxlen = maxlen
    self.with_labels = with_labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent1 = str(self.data.loc[index, 'argument1'])
    sent2 = str(self.data.loc[index, 'argument2'])

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent1, sent2,
                                  add_special_tokens = True,
                                  padding= 'max_length',  # Pad to max_length
                                  truncation= 'longest_first',  # Truncate to max_length
                                  max_length=self.maxlen,
                                  return_attention_mask = True,
                                  return_token_type_ids=True,
                                  return_tensors='pt')  # Return torch.Tensor objects


    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0) # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = float(self.data.loc[index, 'label'])
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids

#_______________________________________________________________________________________________________________________________________
class PairClrBaseModel(nn.Module):
    
  def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
    super(PairClrBaseModel, self).__init__()
    #  Instantiating BERT-based model object
    self.bert_model = bert_model
    self.bert_layer = AutoModel.from_pretrained(bert_model)

    #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
    if bert_model == "albert-base-v2":  # 12M parameters
        hidden_size = 768
    elif bert_model == "bert-base-uncased": # 110M parameters
        hidden_size = 768
    elif bert_model == "distilbert-base-uncased":
        hidden_size = 768

    # Freeze bert layers and only train the classification layer weights
    if freeze_bert:
        for p in self.bert_layer.parameters():
            p.requires_grad = False

    # Classification layer
    self.cls_layer = nn.Linear(hidden_size, 1)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, input_ids, attn_masks, token_type_ids):
    '''
    Inputs:
        -input_ids : Tensor  containing token ids
        -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
        -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
    '''
    if self.bert_model == "distilbert-base-uncased":
      distilbert_output = self.bert_layer(input_ids, attn_masks)
      hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
      pooled_output = hidden_state[:, 0]  # (bs, dim)
      logits = self.cls_layer(self.dropout(pooled_output))
      return logits.squeeze()
    else:
      # Feeding the inputs to the BERT-based model to obtain contextualized representations
      cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

      # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a Linear Layer.
      logits = self.cls_layer(self.dropout(pooler_output))
      return logits
    
class PairClassifierPLSTM(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(PairClassifierPLSTM, self).__init__()
        self.bert_model = bert_model
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(self.bert_model)
        self.hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.lstmEncoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
        self.lstmDecoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)

        # Classification layer
        self.cls_layer = nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        if self.bert_model == "distilbert-base-uncased":
          sequence_output = self.bert_layer(input_ids, attn_masks)[0]
        else:
          sequence_output, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)
        lstmE_output, (h,c) = self.lstmEncoder(sequence_output) ## BiLSTM Encoder
        lstmD_output, (h,c) = self.lstmDecoder(lstmE_output) ## BiLSTM Decoder

        hidden = torch.cat((lstmD_output[:,-1, :int(self.hidden_size/2)],lstmD_output[:,0, int(self.hidden_size/2):]),dim=-1)

        logits = self.cls_layer(self.dropout(hidden.view(-1, self.hidden_size)))

        return logits
#_____________________________________________________________________________________________________________________________________________________
        
class PairClassifierPGRU(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(PairClassifierPGRU, self).__init__()
        self.bert_model = bert_model
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(self.bert_model)
        self.hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.gluEncoder = nn.GRU(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
        self.gluDecoder = nn.GRU(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)

        # Classification layer
        self.cls_layer = nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        if self.bert_model == "distilbert-base-uncased":
          sequence_output = self.bert_layer(input_ids, attn_masks)[0]
        else:
          sequence_output, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)
        gruE_output, _ = self.gluEncoder(sequence_output) ## GRU Encoder
        gruD_output, _ = self.gluDecoder(gruE_output) ## GRU Decoder

        hidden = torch.cat((gruD_output[:,-1, :int(self.hidden_size/2)],gruD_output[:,0, int(self.hidden_size/2):]),dim=-1)

        logits = self.cls_layer(self.dropout(hidden.view(-1, self.hidden_size)))

        return logits
#______________________________________________________________________________________________________________________________________________
class PairClassifierPCNN(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(PairClassifierPCNN, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        self.hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        # .........
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.BN1 = nn.BatchNorm2d(32)
        self.RLU1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.BN2 = nn.BatchNorm2d(64)
        self.RLU2 = nn.ReLU()

        self.MXP2d = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1, stride = 2)
        self.BN3 = nn.BatchNorm2d(128)
        self.RLU3 = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(786432, 1)
        self.SM = nn.Softmax(1)
        

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        
        output, _ = self.bert_layer(input_ids, attn_masks, token_type_ids)
        output = torch.unsqueeze(output, 1)
        output = output.transpose(2, 3)
        
        output = self.conv1(output)
        output = self.BN1(output)
        output = self.RLU1(output)

        output = self.conv2(output)
        output = self.BN2(output)
        output = self.RLU2(output)

        output = self.MXP2d(output)

        output = self.conv3(output)
        output = self.BN3(output)
        output = self.RLU3(output)

        output = self.dropout(output)
        output = output.flatten(1)
        output = self.fc(output)
        #output = self.SM(output)

        return output