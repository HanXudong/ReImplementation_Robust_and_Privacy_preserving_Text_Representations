import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import json
import random

from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import numpy as np

from scipy import optimize

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])



def prediction_transformation(preds, y, age, gender, tag_pad_idx):
    """
    Returns array of pred and tags without padded elements 
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    return (list(max_preds[non_pad_elements].squeeze(1).cpu().numpy()), 
            list(y[non_pad_elements].cpu().numpy()), 
            list(age[non_pad_elements].cpu().numpy()),
            list(gender[non_pad_elements].cpu().numpy()))


def evaluate_bias(model, iterator, tag_pad_idx, return_value = False):
    
    model.eval()
    
    with torch.no_grad():
        t_age = []
        t_gender = []
        preds = []
        gold_labels = []

        for batch in iterator:
            text = batch.text
            tags = batch.tag_label
            gender = batch.gender_label
            age = batch.age_label

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            gender = gender.reshape(text.shape[1],-1).repeat(1,text.shape[0]).T.reshape(-1)
            age = age.reshape(text.shape[1],-1).repeat(1,text.shape[0]).T.reshape(-1)

            t1, t2, t3, t4 = prediction_transformation(predictions, tags, age, gender, tag_pad_idx)

            t_age = t_age + t3
            t_gender = t_gender + t4
            preds = preds + t1
            gold_labels = gold_labels + t2

        t_age = np.array(t_age)
        t_gender = np.array(t_gender)
        preds = np.array(preds)
        gold_labels = np.array(gold_labels)

        acc_overall = accuracy_score(preds, gold_labels)
        acc_O = accuracy_score(preds[t_age==1], gold_labels[t_age==1])
        acc_U = accuracy_score(preds[t_age==0], gold_labels[t_age==0])
        acc_M = accuracy_score(preds[t_gender==1], gold_labels[t_gender==1])
        acc_F = accuracy_score(preds[t_gender==0], gold_labels[t_gender==0])
        
        FScore_overall = f1_score(preds, gold_labels, average='macro')
        FScore_O = f1_score(preds[t_age==1], gold_labels[t_age==1], average='macro')
        FScore_U = f1_score(preds[t_age==0], gold_labels[t_age==0], average='macro')
        FScore_M = f1_score(preds[t_gender==1], gold_labels[t_gender==1], average='macro')
        FScore_F = f1_score(preds[t_gender==0], gold_labels[t_gender==0], average='macro')
        
    if not return_value:
        
        print("Overall")
        print("Accuracy : {:.6f}".format(acc_overall))
        print("F Score : {:.6f}".format(FScore_overall))
        print("Age group")
        print("Accuracy Under35: {:.6f} V.S. Over45: {:.6f}".format(acc_U, acc_O))
        print("F Score Under35: {:.6f} V.S. Over45: {:.6f}".format(FScore_U, FScore_O))
        print("Gender group")
        print("Accuracy Female: {:.6f} V.S. Male: {:.6f}".format(acc_F, acc_M))
        print("F Score Female: {:.6f} V.S. Male: {:.6f}".format(FScore_F, FScore_M))
    
        return 0
    else:
        return {"acc_overall":acc_overall,
                "acc_O":acc_O,
                "acc_U":acc_U,
                "acc_M":acc_M,
                "acc_F":acc_F,
                "FScore_overall":FScore_overall,
                "FScore_O":FScore_O,
                "FScore_U":FScore_U,
                "FScore_M":FScore_M,
                "FScore_F":FScore_F}

    









