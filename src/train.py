#!/usr/bin/env python
# coding: utf-8

import csv
import sys
import os
import torch
import time
import nltk
import datetime
import json
import math
import itertools
import pickle
import umap
import re
import argparse
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, BertForMaskedLM, BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, AutoTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering, pipeline, T5Tokenizer, TFT5Model


from random import *
from tqdm import tqdm
from pylab import *


from copy import deepcopy
from collections import defaultdict

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset", default="wizard", type=str, help="Dataset name."
    )
    parser.add_argument(
        "--MODEL", default="bert-base-uncased", type=str, help="Name of the BERT model to be used as the text encoder."
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch Size."
    )
    parser.add_argument(
        "--max_len", default=32, type=int, help="Maximum length of the sequence after tokenization."
    )
    parser.add_argument(
        "--embedding_size", default=768, type=int, help="Hidden size of the representation output of the text encoder."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="GPU device."
    )
    parser.add_argument(
        "--model_save_path", default="model/wizard/", type=str, help="Save path of the models."
    )
    parser.add_argument(
        "--results_save_path", default="results/wizard/", type=str, help="Save path of the dataset."
    )
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of epochs."
    )
    parser.add_argument(
        "--delta_on", default=False, type=bool
    )
    parser.add_argument(
        "--entropy_on", default=True, type=bool
    )
    parser.add_argument(
        "--save_models", default=True, type=bool
    )
    parser.add_argument(
        "--lambda_1", default=10, type=int, help="lambda 1 value (int)."
    )
    parser.add_argument(
        "--lambda_2", default=1, type=int, help="lambda 2 value (int)."
    )
    parser.add_argument(
        "--seed", default=42, type=int
    )
    return parser

### Utility Functions


def set_seed(args):
    """
    Set the random seed for reproducibility
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def load_dump(filename):
    """Loads any .pkl file and returns the content"""
    with open(filename, "rb") as file:
        return pickle.load(file)


def dump_data(filename, dataset):
    """Dumps data into .pkl file."""
    with open(filename, "wb") as file:
        pickle.dump(dataset, file)

def load(dataset, batch_size, shuffle=True):
    """Loads the dataset using a dataloader with a batch size."""
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

def encode(dataset, tokenizer, max_len):
    """Converts data into token representations in the form of tensors."""
    encoded_dataset = []
    for (sent, label, bias) in tqdm(dataset):
        sent_emb = torch.tensor(tokenizer.encode(sent, max_length = max_len, pad_to_max_length =True, truncation=True, add_special_tokens=True))
        encoded_dataset.append((sent_emb, label, bias))
    return encoded_dataset


def load_funpedia(filename):
    """ Loads funpedia dataset"""

    Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    Z_LABELS = {'male': 0, 'female': 1, 'gender-neutral': 2}

    dataset = []
    with open(filename, "rb") as file:
        lines =  json.load(file)
        for line in lines:
            dataset.append((line[1], Y_LABELS[line[2]], Z_LABELS[line[4]]))
    return dataset

def load_content(content, DATA_IDX = 1, Y_IDX = 2, Z_IDX = 3):
    """
    Forms the data in the format (sentence, y-label, z-label)
    Returns an array of instances in the above format
    """
    dataset = []
    for c in content:
        dataset.append((c[DATA_IDX], c[Y_IDX], c[Z_IDX])) 
    return dataset


def get_dataset(dataset):
    """
    Input: the dataset name args.dataset
    
    Returns: train, dev, test
    train: list of training instances
    dev: list of development instances
    test: list of test instances
    """


    if dataset == "wizard":
        WIZARD_PATH = "../../data/wizard/"
        Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(WIZARD_PATH + "train.pkl"))
        dev = load_content(load_dump(WIZARD_PATH + "dev.pkl"))
        test = load_content(load_dump(WIZARD_PATH + "test.pkl"))

    elif dataset == "convai2":
        CONVAI2_PATH = "data/convai2/"
        Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(CONVAI2_PATH + "train.pkl"))
        dev = load_content(load_dump(CONVAI2_PATH + "dev.pkl"))
        test = load_content(load_dump(CONVAI2_PATH + "test.pkl"))

    elif dataset == "light":
        LIGHT_PATH = "data/light/"
        Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(LIGHT_PATH + "train.pkl"))
        dev = load_content(load_dump(LIGHT_PATH + "dev.pkl"))
        test = load_content(load_dump(LIGHT_PATH + "test.pkl"))

    elif dataset == "opensub":
        OPENSUB_PATH = "data/opensub/"
        Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(OPENSUB_PATH + "train.pkl"))
        dev = load_content(load_dump(OPENSUB_PATH + "dev.pkl"))
        test = load_content(load_dump(OPENSUB_PATH + "test.pkl"))

    elif dataset == "funpedia":
        FUNPEDIA_PATH = "data/funpedia/"
        Y_LABELS = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        Z_LABELS = {'male': 0, 'female': 1, 'gender-neutral': 2}

        train = load_content(load_dump(FUNPEDIA_PATH + "train.pkl"))
        dev = load_content(load_dump(FUNPEDIA_PATH + "dev.pkl"))
        test = load_content(load_dump(FUNPEDIA_PATH + "test.pkl"))

    elif dataset == "dial":
        DIAL_PATH = "data/twitter-race/sentiment-race/"
        Y_LABELS = {'Positive': 0, 'Negative': 1}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(DIAL_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(DIAL_PATH + "test.pkl"))

    elif dataset == "pan16":
        PAN16_PATH = "data/pan16/"
        Y_LABELS = {'Positive': 0, 'Negative': 1}
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(PAN16_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(PAN16_PATH + "test.pkl"))

    elif dataset == "bios":
        BIOS_PATH = "data/bios/"
        Y_LABELS = {}
        for i in range(28):
            Y_LABELS[i] = i
        Z_LABELS = {'male': 0, 'female': 1}

        train = load_content(load_dump(BIOS_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(BIOS_PATH + "test.pkl"))

    return train, dev, test, Y_LABELS, Z_LABELS



### Adversarial Scrubber Components
class Scrubber(nn.Module):
    """
    Scrubber model s(.)
    Architecture: 2 layer MLP with ReLU non-linearity
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.ff1 = nn.Linear(embedding_size, embedding_size)
        self.ff2 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        x_new = self.ff2(F.relu(self.ff1(x)))
        return x_new


class Discriminator(nn.Module):
    """
    Bias Discriminator d(.)
    Architecture: 1 layer MLP
    """

    def __init__(self, num_labels, embedding_size):
        super().__init__()  
        self.linear_1 = nn.Linear(embedding_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        pooled_output = self.dropout(x)
        output = self.linear_1(pooled_output)
        return output


class DeltaLoss:
    """
    \Delta-loss function
    """

    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size(), device)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, device, temperature = 0.1):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    
    def __call__(self, logits, label, device):
        # logits : (batch_size, # output classes in z)
        # label: (batch_size)
        
        gumbled_logits = self.gumbel_softmax(logits, device)
        mask = torch.zeros(logits.size())
                
        for i, m in enumerate(mask):
            m[label[i]] = 1
            
            
        mask = mask.to(device)
        return torch.mean(torch.sum(torch.mul(mask, gumbled_logits), dim=-1))


class EntropyLoss(nn.Module):
    """
    Entropy Loss: -H(x)
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b

# Losses
delta = DeltaLoss()
ce = nn.CrossEntropyLoss()
entropy = EntropyLoss()

def initialize_models(args, Y_LABELS, Z_LABELS, device):
    '''
    Initialize all modules with default parameters
    '''
    bert_model = BertModel.from_pretrained(args.MODEL)
    bert_model.to(device)

    netS = Scrubber(args.embedding_size).to(device)
    print(netS)


    D_bias = Discriminator(len(list(Z_LABELS.keys())), args.embedding_size)
    D_bias.to(device)
    print(D_bias)

    D_task = Discriminator(len(list(Y_LABELS.keys())), args.embedding_size)
    D_task.to(device)
    print(D_task)
    return bert_model, netS, D_bias, D_task


def train_AdS(args, epoch, train_dataloader, bert_model, netS, D_bias, D_task, optimS, optimD_bias, device):
    '''
    Training Adversarial-Scrubber model
    '''

    bert_model.train()
    D_bias.train()
    D_task.train()
    netS.train()
    
    epoch_start_time = time.time()
    
    train_dataloader = tqdm(train_dataloader)
    for i, (data, label, bias) in enumerate(train_dataloader, 0):
        b_size = data.size(0)
        real_data = data.to(device)
        label = label.long().to(device)
        bias = bias.long().to(device)
        
        bert_model.zero_grad()
        netS.zero_grad()
        D_bias.zero_grad()
        D_task.zero_grad()
        
        
        # Updating bias discriminator
        optimD_bias.zero_grad()
        
        bert_output = bert_model(real_data)[1]
        fake_data = netS(bert_output)
        probs_fake = D_bias(fake_data.detach())
        
        loss_fake = ce(probs_fake, bias)
        
        D_loss = loss_fake
        D_loss.backward()
        optimD_bias.step()

        optimS.zero_grad()
        netS.zero_grad()
        D_task.zero_grad()
        bert_model.zero_grad()

        # Fake data treated as real.
        bert_output = bert_model(real_data)[1]
        fake_data = netS(bert_output)

        task_out = D_task(fake_data)
        task_loss =  ce(task_out, label)
        task_loss_ = task_loss.item()
        
        probs_fake = D_bias(fake_data)
        delta_loss = args.lambda_2 * delta(probs_fake, bias, device)
        
        entropy_loss = args.lambda_1 * entropy(probs_fake)
        
        # Scrubber Loss
        S_loss = task_loss 

        if args.entropy_on:
            S_loss -= entropy_loss

        if args.delta_on:
            S_loss += delta_loss

        # Updating Scrubber
        S_loss.backward()
        optimS.step()
        
        train_dataloader.set_description(
            f'D Bias Loss = {(loss_fake.item()):.6f} \
            D Task Loss = {(task_loss_):.6f}\
            G Dirac Loss = {(delta_loss.item()):.6f} \
            Entropy loss = {( entropy_loss.item()):.6f}')

def eval_AdS(epoch, test_loader, bert_model, netS, D_bias, D_task, device):
    bert_model.eval()
    netS.eval()
    D_bias.eval()
    D_task.eval()

    test_loader = tqdm(test_loader)
    task_total_acc = 0
    bias_total_acc = 0
    
    task_total_f1 = 0
    bias_total_f1 = 0
    
    task_total_recall = 0
    bias_total_recall = 0
    
    for step, (data, label, bias) in enumerate(test_loader):
        real_data = data.to(device)
        label = label.long().to(device)
        bias = bias.long().to(device)

        with torch.no_grad():
            bert_output = bert_model(real_data)[1]
            fake_data = netS(bert_output)
            task_logits = D_task(fake_data)
            bias_logits = D_bias(fake_data)
                    
        
        task_logits = task_logits.detach().cpu().numpy()
        task_logits = np.argmax(task_logits, axis=1).flatten()
        label =  label.detach().cpu().numpy().flatten()     
        task_acc = f1_score(label, task_logits, average = 'macro') * 100
        
        
        bias_logits = bias_logits.detach().cpu().numpy()
        bias_logits = np.argmax(bias_logits, axis=1).flatten()
        bias =  bias.detach().cpu().numpy().flatten()     
        bias_acc = f1_score(bias, bias_logits, average = 'macro') * 100
        
        test_loader.set_description(f'task eval acc = {(task_acc.item()):.6f} bias eval acc = {(bias_acc.item()):.6f}')
     
        task_total_acc += task_acc
        bias_total_acc += bias_acc
    
    print(" Task Evaluation Acc: {:.6f}".format(task_total_acc / len(test_loader)))
    print(" Bias Evaluation Acc: {:.6f}".format(bias_total_acc / len(test_loader)))

def save_models(args, bert_model, netS, D_bias, D_task):
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    torch.save(bert_model, os.path.join(args.model_save_path, "bert-model-e+d.pb"))
    torch.save(netS, os.path.join(args.model_save_path, "gen-model-e+d.pb"))
    torch.save(D_bias, os.path.join(args.model_save_path, "disc-bias-e+d.pb"))
    torch.save(D_task, os.path.join(args.model_save_path, "disc-task-e+d.pb"))

def generate_purged_dataset(dataset_loader, bert_model, netS, device):
    dataset = []
    for data, label, bias in tqdm(dataset_loader):
        real_data = data.to(device)
 
        with torch.no_grad():
            bert_output = bert_model(real_data)[1]
            fake_data = bert_output
            fake_data = netS(bert_output)
            
        purged_emb = fake_data.detach().cpu().numpy()
        data_slice = [(data, label, bias) for data, label, bias in zip(purged_emb, label, bias)]
        dataset.extend(data_slice)
    return dataset

def save_representations(args, bert_model, netS, train_dataloader, test_dataloader, device):
    if not os.path.exists(args.results_save_path):
        os.makedirs(args.results_save_path)

    train_dataset = generate_purged_dataset(train_dataloader, bert_model, netS, device)
    test_dataset = generate_purged_dataset(test_dataloader, bert_model, netS, device)

    dump_data(os.path.join(args.results_save_path, 'train-gen.pkl'), train_dataset)
    dump_data(os.path.join(args.results_save_path, 'test-gen.pkl'), test_dataset)

def main():
    args = get_parser().parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # Tokenizer initialize
    tokenizer = BertTokenizer.from_pretrained(args.MODEL)

    # Form dataset
    train, dev, test, Y_LABELS, Z_LABELS = get_dataset(args.dataset)
    train_dataset, dev_dataset, test_dataset = encode(train, tokenizer, args.max_len), encode(dev, tokenizer, args.max_len), encode(test, tokenizer, args.max_len)
    train_dataloader, test_dataloader = load(train_dataset, args.batch_size), load(test_dataset, args.batch_size)

    # Initialize models
    bert_model, netS, D_bias, D_task = initialize_models(args, Y_LABELS, Z_LABELS, device)

    # Task setup, and arguments
    print(Y_LABELS, Z_LABELS)
    print(args)

    optimD_bias = optim.AdamW([{'params': D_bias.parameters()}], lr=2e-5, betas=(0.5, 0.999))
    optimS = optim.AdamW([{'params': netS.parameters()}, {'params': D_task.parameters()} , {'params': bert_model.parameters()}], lr=2e-5, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_AdS(args, epoch, train_dataloader, bert_model, netS, D_bias, D_task, optimS, optimD_bias, device)
        eval_AdS(epoch, test_dataloader, bert_model, netS, D_bias, D_task, device)


    if args.save_models:
        print("Saving models ...")
        save_models(args, bert_model, netS, D_bias, D_task)
        print("Saving representations ...")
        save_representations(args, bert_model, netS, train_dataloader, test_dataloader, device)
        print("Done!")


if __name__ == "__main__":
    main()


