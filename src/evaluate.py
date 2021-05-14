import os
import sys
import pickle
import json
import numpy as np
import argparse
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import svm
from random import *
from tqdm import tqdm
from collections import Counter

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train_path", type=str, default="results/wizard/train-gen.pkl"
    )
    parser.add_argument(
        "--test_path", type=str, default="results/wizard/test-gen.pkl"
    )
    parser.add_argument(
        "--data_id", type=int, default=0
    )
    parser.add_argument(
        "--y_id", type=int, default=1
    )
    parser.add_argument(
        "--z_id", type=int, default=2
    )
    parser.add_argument(
        "--y_classes", type=int, default=3
    )
    parser.add_argument(
        "--z_classes", type=int, default=2
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    return parser

def set_seed(args):
    """
    Set the random seed for reproducibility
    """
    np.random.seed(args.seed)

def load_dump(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class MDL:
    def __init__(self, dataset):
        super(MDL, self).__init__()
        shuffle(dataset)
        
        ratios = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1]
        self.all_datasets = []
        
        for r in ratios:
            self.all_datasets.append(dataset[:int(r * len(dataset))])
    
    def get_score(self, args, num_labels, label_id):
        score = len(self.all_datasets[0]) * math.log(num_labels, 2)
        
        print("Computing MDL ...")
        for i, dataset in tqdm(enumerate(self.all_datasets[:-1]), total=len(self.all_datasets[:-1])):
            X_train = np.array([x[args.data_id] for x in dataset])
            Y_train = [x[label_id] for x in dataset]
                        
            clf = MLPClassifier()
            clf.fit(X_train, Y_train)
            
            next_dataset = self.all_datasets[i+1]
            X_test = np.array([x[args.data_id] for x in next_dataset])
            Y_test = [x[label_id] for x in next_dataset]
            
            Y_pred = clf.predict_proba(X_test)
            
            for y_gold, y_pred in zip(Y_test, Y_pred):
                try:
                    score -= math.log(y_pred[y_gold], 2)
                except:
                    pass
        return score / 1024

def get_performance(args, train, test, label_id):
    X_train = np.array([x[args.data_id] for x in train])
    Y_train = [x[label_id] for x in train]

    X_test = np.array([x[args.data_id] for x in test])
    Y_test = [x[label_id] for x in test]

    clf = MLPClassifier()


    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    F1 = f1_score(y_pred, Y_test, average='macro') * 100
    P = precision_score(y_pred, Y_test, average='macro') * 100
    R = recall_score(y_pred, Y_test, average='macro') * 100

    return F1, P, R


def main():
    parser = get_config()
    args = parser.parse_args()

    set_seed(args)

    train, test = load_dump(args.train_path), load_dump(args.test_path)

    f1, p, r = get_performance(args, train, test, args.y_id)
    print("Y Classification Performance: F1 - {} P - {} R - {}".format(f1, p, r))

    f1, p, r = get_performance(args, train, test, args.z_id)
    print("Z Classification Performance: F1 - {} P - {} R - {}".format(f1, p, r))

    # MDL
    mdl = MDL(train)
    print("MDL for Y: {}".format(mdl.get_score(args, args.y_classes, 1)))
    print("MDL for Z: {}".format(mdl.get_score(args, args.z_classes, 2)))

if __name__ == "__main__":
    main()





