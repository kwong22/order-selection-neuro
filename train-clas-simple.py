#!/usr/bin/env python

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

import os
import time
from tqdm import tqdm
import wandb

from utils import *
import model.lstm_clas_with_meta_simple

# Config
PROJECT_NAME = 'order-clas-simple'
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
CORRECTIONS_PATH = 'data/misspellings.txt'
BIOWORDVEC_PATH = 'data/bio_embedding_extrinsic'
LM_PATH = 'checkpoints/lm-weights1.pth' # Load language model weights from here
CLAS_PATH = 'checkpoints/clas-with-meta-simple-weights1.pth' # Save classification model weights to here

hyperparameter_defaults = {
    'epochs': 1,
    'batch_size': 64,
    'lstm_hidden_size': 256,
    'lstm_num_layers': 1,
    'fc_hidden_size': 256,
    'dropout': 0.2,
    'freeze_embeds': True,
    'optimizer': 'adam',
    'learning_rate': 1e-7,
}

wandb.init(config=hyperparameter_defaults, project=PROJECT_NAME)
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', torch.cuda.get_device_name())


def evaluate(model, dataloader, loss_fn, acc_fn, weight=None):
    model.eval() # set to eval mode
    
    total_loss = 0
    total_acc = 0
    total_count = 0

    with torch.no_grad(): # do not compute gradients during evaluation
        for idx, (text, label) in enumerate(dataloader):
            pred_label = model(text)
            total_loss += loss_fn(pred_label, label, weight).item()
            total_acc += acc_fn(pred_label, label)
            total_count += label.shape[0] # adds batch size
    return total_loss/total_count, total_acc/len(dataloader)


def train(
    model,
    train_dataloader,
    valid_dataloader,
    loss_fn,
    optimizer,
    acc_fn,
    weight=None,
    num_epochs=1,
    save_interval=None,
):
    example_count = 0
    batch_count = 0
    
    train_losses = []
    valid_losses = []
    valid_idx = [] # track index relative to train losses for plotting
    
    pbar = tqdm() # monitor number of batches completed within epoch
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training
        model.train() # set to train mode

        total_train_loss = 0
        total_train_count = 0
        
        pbar.reset(total=len(train_dataloader)) # reset and reuse the bar
        pbar.set_description(desc='epoch {}/{}'.format(epoch, num_epochs))
        
        for idx, (text, label) in enumerate(train_dataloader):
            # Forward pass
            out = model(text)
            loss = loss_fn(out, label, weight) # calculate loss
            
            # Backward pass
            optimizer.zero_grad() # reset gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # clip gradients
            
            # Step with optimizer
            optimizer.step() # adjust parameters by gradients collected in backward pass
            
            example_count += len(label)
            batch_count += 1
            
            train_losses.append(loss.item() / len(label))
            total_train_loss += loss.item()
            total_train_count += len(label) # adds batch size
            
            pbar.update() # increment bar by 1
        
        # Validation
        valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn, acc_fn, weight)
        valid_losses.append(valid_loss)
        valid_idx.append(len(train_losses) - 1)

        metrics = {
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }

        wandb.log(metrics)

        print('\nepoch {:3d}: valid_loss: {:8.6f}, valid_acc: {:8.6f}'.format(
            epoch,
            valid_loss,
            valid_acc,
            )
        )
    
    pbar.close()


def main():
    train_ds = load_dataset(TRAIN_PATH)
    print('loaded train set')

    test_ds = load_dataset(TEST_PATH)
    print('loaded test set')

    corrections = load_corrections(CORRECTIONS_PATH)
    vocab = build_vocab_with_corrections(train_ds, my_tokenizer, corrections)
    print('built vocab')

    load_biowordvec_embeddings(BIOWORDVEC_PATH, vocab)
    print('loaded embeddings')


    # Text pipeline returns a list of tokens
    text_pipeline = lambda x: [vocab[token] for token in my_tokenizer(x)]

    # Encode label
    label_names = [
        'CT HEAD WITHOUT CONTRAST',
        'CT HEAD WITH CONTRAST',
        'MRI BRAIN WITHOUT CONTRAST',
        'MRI BRAIN WITH CONTRAST',
    ]
    encode_label = {x: i for i, x in enumerate(label_names)}
    label_pipeline = lambda x: encode_label[x]

    # Encode sex
    sex_names = [
        'Male',
        'Female',
    ]
    encode_sex = {x: i for i, x in enumerate(sex_names)}
    sex_pipeline = lambda x: encode_sex[x]

    # Encode age
    age_pipeline = lambda x: x

    # Encode patient class
    pt_class_names = [
        'Inpatient',
        'Emergency',
        'Outpatient',
        'None',
    ]
    encode_pt_class = {x: i for i, x in enumerate(pt_class_names)}
    def pt_class_pipeline(x):
        if pd.isna(x):
            return encode_pt_class['None']
        return encode_pt_class[x]

    # Encode GFR
    gfr_names = [
        '<30',
        '30-60',
        '>60',
        'None',
    ]
    encode_gfr = {x: i for i, x in enumerate(gfr_names)}
    def gfr_pipeline(x):
        if pd.isna(x):
            return encode_gfr['None']
        return encode_gfr[x]


    def clas_collate_batch(batch):
        metadata_list = []
        text_list = []
        label_list = []
        len_list = []
        for metadata, text, label in batch:
            if not pd.isna(text):
                processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
                if len(processed_text) > 0:
                    sex, age, pt_class, gfr = metadata
                    sex_tensor = nn.functional.one_hot(
                        torch.tensor(sex_pipeline(sex)),
                        num_classes=len(sex_names)
                    )
                    
                    age_tensor = torch.tensor(age_pipeline(age)).unsqueeze(dim=0)
                    
                    pt_class_tensor = nn.functional.one_hot(
                        torch.tensor(pt_class_pipeline(pt_class)),
                        num_classes=len(pt_class_names)
                    )
                    
                    gfr_tensor = nn.functional.one_hot(
                        torch.tensor(gfr_pipeline(gfr)),
                        num_classes=len(gfr_names)
                    )
                    
                    metadata_list.append(torch.cat([
                        sex_tensor,
                        age_tensor,
                        pt_class_tensor,
                        gfr_tensor,
                    ]))
                    
                    text_list.append(processed_text)
                    len_list.append(len(processed_text))
                    label_list.append(label_pipeline(label))
        
        # pack_padded_sequence expects lens sorted in decreasing order
        len_order = np.flip(np.argsort(len_list)) # sorted in ascending, then flip
        metadata_list = [metadata_list[i] for i in len_order]
        text_list = [text_list[i] for i in len_order]
        label_list = [label_list[i] for i in len_order]
        len_list = [len_list[i] for i in len_order]
        
        pad_idx = 1 # vocab.stoi['<pad>'] = 1
        
        metadata_list = torch.stack(metadata_list, dim=0)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        len_list = torch.tensor(len_list, dtype=torch.int64)
        
        return (metadata_list.to(device), text_list.to(device)), label_list.to(device)

    # Hyperparameters
    EPOCHS = config.epochs
    LR = config.learning_rate
    BATCH_SIZE = config.batch_size

    # Split dataset
    num_train = int(len(train_ds) * 0.8)
    split_train_, split_valid_ = random_split(train_ds, [num_train, len(train_ds) - num_train])

    train_dl = DataLoader(split_train_, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=clas_collate_batch)
    valid_dl = DataLoader(split_valid_, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=clas_collate_batch)
    test_dl = DataLoader(split_valid_, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=clas_collate_batch)

    clas_model = model.lstm_clas_with_meta_simple.LSTMClassifier(
        len(vocab),
        vocab.vectors.shape[1],
        config.freeze_embeds,
        config.lstm_hidden_size,
        config.lstm_num_layers,
        len(sex_names) + 1 + len(pt_class_names) + len(gfr_names),
        config.fc_hidden_size,
        config.dropout,
        len(label_names),
        vocab.vectors,
    ).to(device)

    # Load all but last layer
    load_encoder(LM_PATH, clas_model, last_layer_name='hidden2out')

    wandb.watch(clas_model)

    OPTIM = config.optimizer
    if OPTIM == 'adam':
        clas_optimizer = torch.optim.Adam(clas_model.parameters(), lr=LR)
    elif OPTIM == 'sgd':
        clas_optimizer = torch.optim.SGD(clas_model.parameters(), lr=LR)

    class_weights = compute_class_weights(train_ds, label_names).to(device)

    train(
        clas_model,
        train_dl,
        valid_dl,
        model.lstm_clas_with_meta_simple.loss_fn,
        clas_optimizer,
        model.lstm_clas_with_meta_simple.accuracy,
        weight=class_weights,
        num_epochs=EPOCHS,
        )

    # Save model weights
    save_checkpoint(CLAS_PATH, clas_model, clas_optimizer)


if __name__ == '__main__':
    main()
