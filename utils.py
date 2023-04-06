import numpy as np
import os
import re

from collections import Counter
import pandas as pd
import torch
from torchtext.vocab import Vocab

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import KeyedVectors


def prepare_train_set(path):
    df = pd.read_excel(path)
    
    col_names = [
        'id',
        'proc',
        'order_dx',
        'reason',
        'auth_prov',
        'dept',
        'order_comments',
        'protocol',
        'order_info',
        'sex',
        'age',
        'proc_name',
        'pt_class',
        'performing_dept',
        'lab_name',
        'gfr',
    ]
    df.columns = col_names

    # Data pre-processing
    pt_class_replacements = {
        'Emergency': 'Emergency',
        'Inpatient': 'Inpatient',
        'Outpatient': 'Outpatient',
        'Observation': 'Emergency',
        'Hospital Ambulatory Surgery': 'Outpatient',
        'Case Management': 'Inpatient',
        'Surgery Admit': 'Inpatient',
        'Series': 'Outpatient',
    }
    df['pt_class'] = df['pt_class'].map(pt_class_replacements, na_action=None)

    proc_replacements = {
        'CT HEAD WITHOUT CONTRAST': 'CT HEAD WITHOUT CONTRAST',
        'CT HEAD WITH CONTRAST': 'CT HEAD WITH CONTRAST',
        'CT HEAD WITH AND WITHOUT CONTRAST': 'CT HEAD WITH CONTRAST',
        'MRI BRAIN WITHOUT CONTRAST': 'MRI BRAIN WITHOUT CONTRAST',
        'MRI BRAIN WITH CONTRAST': 'MRI BRAIN WITH CONTRAST',
        'MRI BRAIN WITH AND WITHOUT CONTRAST': 'MRI BRAIN WITH CONTRAST'
    }
    df['label'] = df['proc_name'].map(proc_replacements, na_action=None)

    def gfr_map(val):
        if pd.isna(val):
            return None

        if val == '>60':
            return '>60'

        if val.isnumeric():
            num = int(val)
            if num < 30:
                return '<30'
            elif num <= 60:
                return '30-60'
            else:
                return '>60'

        return None

    df['gfr'] = df['gfr'].map(gfr_map, na_action=None)
    
    def concat_text(row):
        cols = ['order_dx', 'reason', 'order_comments']
    
        # Exclude NA entries
        fragments = [row[c] for c in cols if not pd.isna(row[c])]

        return (' '.join(fragments)).strip()

    df['text'] = df.apply(concat_text, axis=1)
    
    desired_cols = [
        'id',
        'sex',
        'age',
        'pt_class',
        'gfr',
        'text',
        'label',
    ]
    
    return df[desired_cols]


def prepare_test_set(path):
    df = pd.read_excel(path)
    
    col_names = [
        'random',
        'id',
        'auth_prov',
        'dept',
        'reason',
        'sex',
        'age',
        'pt_class',
        'lab_name',
        'gfr',
        'modality',
        'contrast',
    ]
    df.columns = col_names
    
    # Data pre-processing
    pt_class_replacements = {
        'Emergency': 'Emergency',
        'Inpatient': 'Inpatient',
        'Outpatient': 'Outpatient',
        'Observation': 'Emergency',
        'Hospital Ambulatory Surgery': 'Outpatient',
        'Case Management': 'Inpatient',
        'Surgery Admit': 'Inpatient',
        'Series': 'Outpatient',
    }
    df['pt_class'] = df['pt_class'].map(pt_class_replacements, na_action=None)
    
    def get_label(row):
        mod = row['modality'].strip()
        con = row['contrast'].strip()

        if mod == 'CT':
            if con == 'NC':
                return 'CT HEAD WITHOUT CONTRAST'
            elif con == 'C':
                return 'CT HEAD WITH CONTRAST'
            else:
                return None
        elif mod == 'MR':
            if con == 'NC':
                return 'MRI BRAIN WITHOUT CONTRAST'
            elif con == 'C':
                return 'MRI BRAIN WITH CONTRAST'
            else:
                return None

        return None

    df['label'] = df.apply(get_label, axis=1)
    
    def gfr_map(val):
        if pd.isna(val):
            return None

        if val == '>60':
            return '>60'

        if val.isnumeric():
            num = int(val)
            if num < 30:
                return '<30'
            elif num <= 60:
                return '30-60'
            else:
                return '>60'

        return None

    df['gfr'] = df['gfr'].map(gfr_map, na_action=None)
    
    def process_text(s):
        if pd.isna(s):
            return None

        return s.strip()

    df['text'] = df['reason'].apply(process_text)
    
    desired_cols = [
        'id',
        'sex',
        'age',
        'pt_class',
        'gfr',
        'text',
        'label',
    ]
    
    return df[desired_cols]


def load_dataset(path):
    """
    Loads dataframe and converts into list of tuples
    """
    df = pd.read_csv(path)

    def create_tuple(row):
        return ((row['sex'], row['age'], row['pt_class'], row['gfr']),
                row['text'],
                row['label'],
                )
    
    return df.apply(create_tuple, axis=1).tolist()


def normalize_text(s):
    """
    Converts text to lowercase, replaces certain terms, removes ICD codes,
    removes punctuation, removes numbers, and removes extra whitespace.

    Args:
        s (string): text to be normalized

    Returns:
        normalized text
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_icd(text):
        # Remove ICD9 codes (brackets and everything in between them)
        return re.sub(r'\[.*?\]', '', text)
    
    def remove_punc(text):
        punc_no_space = set('\'') # replace with nothing
        punc_space = set('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~') # replace with space
        # keep hyphens
                
        out = []
        
        for c in text:
            if c in punc_no_space:
                continue # replace with nothing
            elif c in punc_space:
                out.append(' ') # replace with space
            else:
                out.append(c) # keep current character
        
        return ''.join(out)
    
    def remove_numbers(text):
        return re.sub(r'\d+', '', text)
        
    def replace_known(text):
        known_words = {
            'r/o': 'ruleout',
            'y/o': 'yearold',
            'yo': 'yearold',
            'y.o.': 'yearold',
            's/p': 'statuspost',
            'h/o': 'historyof',
            'hx': 'history',
            'n/v': 'nausea vomiting',
            'f/u': 'followup',
            'p/w': 'presentedwith',
            'sx': 'symptom',
            'w/u': 'workup',
        }
        
        tokens = text.split()
        
        out = []
        
        for token in tokens:
            if token in known_words:
                out.append(known_words[token]) # replace matching token
            else:
                out.append(token) # keep current token
        
        return ' '.join(out)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_numbers(remove_punc(remove_icd(replace_known(lower(s))))))


def my_tokenizer(s):
    """Normalizes and splits text into tokens.

    Args:
        s (string): text to be tokenized

    Returns:
        list of normalized tokens
    """
    return normalize_text(s).split(' ')


def build_vocab(dataset, tokenizer, vectors=None):
    """Builds vocabulary from all text information in a dataset.

    Args:
        dataset: list of tuples (text, label)
        tokenizer: function that splits text into tokens
        vectors: pre-trained vectors (optional)
            specified in https://pytorch.org/text/0.9.0/vocab.html

    Returns:
        a torchtext.vocab.Vocab object
    """
    counter = Counter()

    for _, text, _ in dataset:
        if not pd.isna(text):
            counter.update(tokenizer(text))

    return Vocab(counter, min_freq=1, vectors=vectors)


def load_corrections(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line.split())

    res = {}
    
    # Exclude first line, which is number of lines in file
    for line in lines[2:]:
        value = ''
        if line[-1].isnumeric():
            offset = int(line[-1])
            if offset > 0:
                # change to one of the other options
                temp = line[line.index('->') + offset]
                # '~' for space
                value = temp.replace('~', ' ')
            else:
                # keep same term
                value = line[1]
        else:
            if line[-1] == '#':
                # '#' for deletion
                value = ''
            else:
                # choose first option
                if len(line) > line.index('->') + 1:
                    temp = line[line.index('->') + 1]
                    # '~' for space
                    value = temp.replace('~', ' ')
                else:
                    # keep same term
                    value = line[1]
        
        res[line[1]] = value
    
    return res


def tokenize_with_corrections(s, tokenizer, corrections):
    tokens = tokenizer(s)
        
    # Replaced out-of-vocabulary tokens
    for i in range(len(tokens)):
        if tokens[i] in corrections:
            tokens[i] = corrections[tokens[i]]

    # Resolve whitespace
    tokens = ' '.join(tokens).split()
    
    return tokens


def build_vocab_with_corrections(dataset, tokenizer, corrections, vectors=None):
    counter = Counter()

    for _, text, _ in dataset:
        if not pd.isna(text):
            counter.update(tokenize_with_corrections(text, tokenizer, corrections))

    return Vocab(counter, min_freq=1, vectors=vectors)


def load_biowordvec_embeddings(path, vocab):
    embed_dim = 200 # set by BioWordVec
    vocab_size = len(vocab.itos)
    
    count = 0
    
    embeds = np.zeros((vocab_size, embed_dim))
    
    vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    
    for idx in range(vocab_size):
        word = vocab.itos[idx]
        
        if word in vectors:
            count += 1
            vector = vectors.get_vector(word)
            embeds[idx] = vector
        else:
            pass
            #print(word)
    
    embeds = torch.from_numpy(embeds).float()
    vocab.set_vectors(vocab.stoi, embeds, embed_dim)
    
    #print('{} / {}'.format(count, len(vocab)))


def compute_class_weights(dataset, label_names):
    # From sklearn.utils.class_weight.compute_class_weight
    encode_label = {x: i for i, x in enumerate(label_names)}
    labels = [encode_label[label] for _, _, label in dataset]
    return torch.tensor(len(labels) / (len(label_names) * np.bincount(labels))).float()


def save_checkpoint(checkpoint_path, model, optimizer=None):
    """Saves model and training parameters at checkpoint_path

    Args:
        checkpoint_path: (string) filename to be saved to
        model: (torch.nn.Module) model containing parameters to save
        optimizer: (torch.optim) optimizer containing parameters to save
    """
    state = {}
    state['model_state_dict'] = model.state_dict()
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(state, checkpoint_path)

    print('model saved to {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model parameters (state_dict) from checkpoint_path

    Args:
        checkpoint_path: (string) filename to be loaded from
        model: (torch.nn.Module) model for which parameters are loaded
        optimizer: (torch.optim) optimizer for which parameters are loaded
    """
    if not os.path.exists(checkpoint_path):
        raise('File does not exist {}'.format(checkpoint_path))

    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    print('model loaded from {}'.format(checkpoint_path))


def load_encoder(checkpoint_path, model, last_layer_name):
    """Loads model parameters for all except last layer from checkpoint_path

    Args:
        checkpoint_path: (string) filename to be loaded from
        model: (torch.nn.Module) model for which parameters are loaded
        last_layer_name: (string) name of layer to exclude
    """
    checkpoint = torch.load(checkpoint_path)

    # Extract states to load (all but last layer)
    states_to_load = {}
    for name, param in checkpoint['model_state_dict'].items():
        if not name.startswith(last_layer_name):
            states_to_load[name] = param

    # Copy current model state and update specified states
    model_state = model.state_dict()
    model_state.update(states_to_load)

    # Replace current model state
    model.load_state_dict(model_state)
    print('model loaded from {}'.format(checkpoint_path))
