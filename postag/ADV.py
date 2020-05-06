from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import json
import random

from tqdm import tqdm, trange


from XD.generate_hyperparameter import *
from XD.POS_utils import *

# Dataset path
TrustPilot_processed_dataset_path = "..//..//dataset//TrustPilot_processed//"
WebEng_processed_dataset_path = "..//..//dataset//Web_Eng//"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'Models/EVP_ADV_{}.pt'

# ####################################
# #         Hyper-parameters         #
# ####################################
# BATCH_SIZE = 64
# LEARNING_RATE = 1e-3
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# N_LAYERS = 2
# BIDIRECTIONAL = True
# DROPOUT = 0.25
# NUM_EPOCHS = 100
# SEED = 960925
# MIN_FREQ = 2
# SAMPLING_INDEX = 10
# LAMBDA = 1e-3

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
    
    def hidden_state(self, text):

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
        
        
        # hidden state = [batch size, sent len * hid dim * n directions]
        # hs = torch.stack([torch.squeeze(i.reshape((outputs.shape[0]*outputs.shape[2],-1))) for i in torch.torch.chunk(outputs, outputs.shape[1], 1)], dim=0)
        return torch.mean(outputs, 0, keepdim=False)

####################################
#      Build the Discriminator     #
####################################
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        # features = features.view(x.shape[0], -1)
        out = F.log_softmax(out, dim=1)
        return out


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.tag_label
        
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_ADV(model,
        discriminator_age,
        discriminator_gender,
        iterator,
        optimizer,
        optimizer_age,
        optimizer_gender,
        criterion,
        criterion_age,
        criterion_gender,
        tag_pad_idx,
        LAMBDA):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.tag_label
        
        
        h = model.hidden_state(text)
        
        """
        Update age discriminitor
        """
        y_age = batch.age_label
        
        y_age_pred = discriminator_age(h).squeeze()
        
        age_loss = LAMBDA * criterion_age(y_age_pred, y_age)
        
        optimizer_age.zero_grad()
        age_loss.backward(retain_graph=True)
        optimizer_age.step()
        
        """
        Update gender discriminitor
        """
        y_gender = batch.gender_label
        y_gender_pred = discriminator_gender(h).squeeze()
        
        gender_loss = LAMBDA * criterion_gender(y_gender_pred, y_gender)
        
        optimizer_gender.zero_grad()
        gender_loss.backward(retain_graph=True)
        optimizer_gender.step()
        
        
        """
        Update Tagger
        """
        y_gender_pred = discriminator_gender(h).squeeze()
        
        gender_loss = criterion_gender(y_gender_pred, y_gender)
        
        y_age_pred = discriminator_age(h).squeeze()
        
        age_loss = criterion_age(y_age_pred, y_age)


        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags) - LAMBDA * (age_loss + gender_loss)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.tag_label
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def ADV_model(BATCH_SIZE = 64, 
            LEARNING_RATE = 1e-3, 
            EMBEDDING_DIM = 100, 
            HIDDEN_DIM = 256,
            N_LAYERS = 2, 
            BIDIRECTIONAL = True, 
            DROPOUT = 0.25,
            NUM_EPOCHS = 100,
            SEED = 960925,
            MIN_FREQ = 2,
            SAMPLING_INDEX = 10,
            LAMBDA=1e-3):
    
    hyperparameters = {
        "BATCH_SIZE" : BATCH_SIZE, 
        "LEARNING_RATE" : LEARNING_RATE, 
        "EMBEDDING_DIM" : EMBEDDING_DIM, 
        "HIDDEN_DIM" : HIDDEN_DIM,
        "N_LAYERS" : N_LAYERS, 
        "BIDIRECTIONAL" : BIDIRECTIONAL, 
        "DROPOUT" : DROPOUT,
        "NUM_EPOCHS" : NUM_EPOCHS,
        "SEED" : SEED,
        "MIN_FREQ" : MIN_FREQ,
        "SAMPLING_INDEX" : SAMPLING_INDEX,
        "LAMBDA" : LAMBDA
    }
    
    print(hyperparameters)
    
    writer = SummaryWriter("EVP/ADV/ADV_{}".format(SAMPLING_INDEX))
    
    # logging hyperparameters
    # writer.add_custom_scalars(hyperparameters)
    for hp in hyperparameters.keys():
        writer.add_scalar("Hyperparameter/"+hp, hyperparameters[hp])
    
    ## Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ####################################
    #          Preparing Data          #
    ####################################

    # 1. data.Field()
    TEXT = data.Field(lower = True)
    TAG_LABEL = data.Field(unk_token = None)
    AGE_LABEL = data.LabelField()
    GENDER_LABEL = data.LabelField()

    fields = {'text':('text', TEXT), 
            'tag_label':('tag_label', TAG_LABEL),
            'age_label':('age_label', AGE_LABEL),
            'gender_label':('gender_label', GENDER_LABEL)}

    # data.TabularDataset
    train_data, valid_data, test_data = data.TabularDataset.splits(path=TrustPilot_processed_dataset_path,
                                                                train="train.jsonl",
                                                                validation = "valid.jsonl",
                                                                test="test.jsonl",
                                                                fields=fields,
                                                                format="json")


    we_train_data, we_valid_data, we_test_data = data.TabularDataset.splits(path=WebEng_processed_dataset_path,
                                                                train="train.jsonl",
                                                                validation = "valid.jsonl",
                                                                test="test.jsonl",
                                                                fields=fields,
                                                                format="json")

    # data.BucketIterator
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                batch_size=BATCH_SIZE,
                                                                device=device,
                                                                sort_key=lambda x: len(x.text))


    we_train_iter, we_valid_iter, we_test_iter = data.BucketIterator.splits((we_train_data, we_valid_data, we_test_data),
                                                                batch_size=BATCH_SIZE,
                                                                device=device,
                                                                sort_key=lambda x: len(x.text))


    # 5. Build vocab
    # TEXT.build_vocab(train_data)
    TAG_LABEL.build_vocab(train_data)
    AGE_LABEL.build_vocab(train_data)
    GENDER_LABEL.build_vocab(train_data)

    TEXT.build_vocab(we_train_data, 
                    min_freq = MIN_FREQ,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)


    # Parameters
    INPUT_DIM = len(TEXT.vocab)
    TAG_OUTPUT_DIM = len(TAG_LABEL.vocab)
    AGE_OUTPUT_DIM = len(AGE_LABEL.vocab)
    GENDER_OUTPUT_DIM = len(GENDER_LABEL.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # Create an Age Discriminator instance
    discriminator_age = Discriminator(input_size=HIDDEN_DIM*2,
                                    hidden_size=HIDDEN_DIM,
                                    num_classes=2)

    discriminator_age.to(device)

    criterion_age = nn.CrossEntropyLoss().to(device)
    optimizer_age = optim.Adam(discriminator_age.parameters(), lr=LEARNING_RATE)

    # Create an Gender Discriminator instance
    discriminator_gender = Discriminator(input_size=HIDDEN_DIM*2,
                                    hidden_size=HIDDEN_DIM,
                                    num_classes=2)

    discriminator_gender.to(device)

    criterion_gender = nn.CrossEntropyLoss().to(device)
    optimizer_gender = optim.Adam(discriminator_gender.parameters(), lr=LEARNING_RATE)


    model = BiLSTMPOSTagger(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            TAG_OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)

    model.apply(init_weights)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)


    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    TAG_PAD_IDX = TAG_LABEL.vocab.stoi[TAG_LABEL.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    best_epoch = -1
        
    saved_model = model_name.format(SAMPLING_INDEX+1)
        
    for epoch in trange(NUM_EPOCHS):    
        train_loss, train_acc = train(model, we_train_iter, optimizer, criterion, TAG_PAD_IDX)
        valid_loss, valid_acc = evaluate(model, we_valid_iter, criterion, TAG_PAD_IDX)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), saved_model)

    model.load_state_dict(torch.load(saved_model))

    best_valid_loss = float('inf')
                
    for epoch in trange(NUM_EPOCHS):    
        # train_loss, train_acc = train(model, train_iter, optimizer, criterion, TAG_PAD_IDX)
        train_loss, train_acc = train_ADV(model,
                                        discriminator_age,
                                        discriminator_gender,
                                        train_iter,
                                        optimizer,
                                        optimizer_age,
                                        optimizer_gender,
                                        criterion,
                                        criterion_age,
                                        criterion_gender,
                                        TAG_PAD_IDX,
                                        LAMBDA)
        # valid_loss, valid_acc = evaluate(model, valid_iter, criterion, TAG_PAD_IDX)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        valid_evaluation = evaluate_bias(model = model, 
                                        iterator = valid_iter,
                                        tag_pad_idx = TAG_PAD_IDX,
                                        return_value = True)

        # print(valid_evaluation)

        for metric in valid_evaluation.keys():
            writer.add_scalar(metric+'/valid', valid_evaluation[metric], epoch)
    # writer.flush()
    writer.close()

if __name__ == "__main__":
    # for si in trange(100):
    #     BATCH_SIZE = int(choice_one([16, 32, 64, 128, 256]))
    #     LEARNING_RATE = log_uniform(-5, -1)
    #     EMBEDDING_DIM = 100
    #     HIDDEN_DIM = int(choice_one([int(i*64) for i in range(1, 9)]))
    #     N_LAYERS = 2
    #     BIDIRECTIONAL = True
    #     DROPOUT = np.random.uniform(0, 0.5)
    #     NUM_EPOCHS = 100
    #     SEED = random_seed()
    #     MIN_FREQ = 2
    #     SAMPLING_INDEX = si
    #     LAMBDA = log_uniform(-5, +2)

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    NUM_EPOCHS = 100
    SEED = 960925
    MIN_FREQ = 2
    SAMPLING_INDEX = 10
    LAMBDA = 1e-3

    ADV_model(BATCH_SIZE = BATCH_SIZE, 
            LEARNING_RATE = LEARNING_RATE,
            EMBEDDING_DIM = EMBEDDING_DIM,
            HIDDEN_DIM = HIDDEN_DIM,
            N_LAYERS = N_LAYERS,
            BIDIRECTIONAL = BIDIRECTIONAL,
            DROPOUT = DROPOUT,
            NUM_EPOCHS = NUM_EPOCHS,
            SEED = SEED,
            MIN_FREQ = MIN_FREQ,
            SAMPLING_INDEX = SAMPLING_INDEX,
            LAMBDA = LAMBDA)
        torch.cuda.empty_cache()