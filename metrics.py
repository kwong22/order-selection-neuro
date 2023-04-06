import itertools
import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt


def get_confusion_matrix(model, dataloader, target_names, thresh=0, k=1, return_cds=False):
    model.eval() # set to eval mode

    softmax = torch.nn.Softmax(dim=1)
    
    # Actual on axis 0, predicted on axis 1
    cm_auto = np.zeros((len(target_names), len(target_names)), dtype=int)
    cm_cds = np.zeros((len(target_names), len(target_names)), dtype=int)

    with torch.no_grad(): # do not compute gradients during evaluation
        #for idx, (text, label, lens) in enumerate(dataloader):
        for idx, (text, label) in enumerate(dataloader):
            pred_label = model(text)
            
            probs = softmax(pred_label) # convert to probability distribution
            sorted_probs = torch.argsort(probs, dim=1, descending=True) # sort indices by probabilities in descending order
            
            for i in range(label.shape[0]):
                if probs[i][sorted_probs[i][0]] > thresh:
                    # if top probability is greater than threshold, then model is confident
                    # automatic mode
                    cm_auto[label[i], sorted_probs[i][0]] += 1
                else:
                    # model is not confident, consider other predictions
                    # clinical decision support mode
                    # consider top k probabilities
                    # assume 0 (no match) or 1 (match) since argsort outputs unique indices
                    if (sorted_probs[i, :k] == label[i]).sum() > 0:
                        cm_cds[label[i], label[i]] += 1 # mark as correct (correct label was in top k probabilities)
                    else:
                        cm_cds[label[i], sorted_probs[i][0]] += 1 # mark as incorrect (as if model chose top probability)
    
    if return_cds:
        return cm_auto, cm_cds
    else:
        return cm_auto + cm_cds


def plot_confusion_matrix_from_array(conf_mat, target_names, plot_txt=True):
    # This function is mostly copied from the sklearn docs
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names, rotation=0)

    if plot_txt:
        thresh = conf_mat.max() / 2.
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            coeff = f'{conf_mat[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.grid(False)


def plot_confusion_matrix(model, dataloader, target_names, plot_txt=True):
    conf_mat = get_confusion_matrix(model, dataloader, target_names)

    plot_confusion_matrix_from_array(conf_mat, target_names, plot_txt)


def calculate_metrics_from_cm(cm, target_names, print_metrics=True):
    tp = np.array([cm[i, i] for i in range(len(cm))])
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp

    tot = cm.sum()
    tn = tot - tp - fn - fp

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    f1 = 2 * sens * ppv / (sens + ppv)

    if print_metrics:
        metrics = pd.DataFrame(np.stack([target_names, sens, spec, ppv, npv, f1], axis=1),
                            columns=['label', 'sensitivity (recall)', 'specificity', 'ppv (precision)', 'npv', 'f1 score'])
        display(metrics)

    weights = cm.sum(axis=1) / tot # weighted by frequency of true label in dataset
    weighted_ppv = np.sum(weights * ppv)
    weighted_sens = np.sum(weights * sens) # same as overall accuracy
    weighted_f1 = np.sum(weights * f1)

    return weighted_ppv, weighted_sens, weighted_f1


def calculate_metrics(model, dataloader, target_names, print_metrics=True):
    cm = get_confusion_matrix(model, dataloader, target_names)
    return calculate_metrics_from_cm(cm, target_names, print_metrics)


def top_losses(model, dataloader, loss_fn, vocab, target_names, weight=None, k=10):
    model.eval() # set to eval mode

    tuples = []

    with torch.no_grad(): # do not compute gradients during evaluation
        #for idx, (text, label, lens) in enumerate(dataloader):
        for idx, (text, label) in enumerate(dataloader):
            pred_label = model(text)

            batch_size = len(pred_label)
            for i in range(batch_size):
                if pred_label[i].argmax() != label[i]:
                    loss = loss_fn(pred_label[i].unsqueeze(0), label[i].unsqueeze(0), weight)
                    converted_text = ' '.join([vocab.itos[idx.item()] for idx in text[i]])
                    tuples.append((converted_text, target_names[pred_label[i].argmax()], target_names[label[i]], loss.item()))

    tuples.sort(key=lambda a: a[3], reverse=True)

    losses = pd.DataFrame(tuples[:k], columns=['text', 'pred_label', 'label', 'loss'])

    display(losses)
