import os
import torch
import pickle
import argparse


from transformers import BertModel, BertTokenizer
from datasets import load_dataset

from random import shuffle
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="wizard")
    parser.add_argument(
        "--MODEL",
        default="bert-base-uncased",
        type=str,
        help="Name of the BERT model to be used as the text encoder.",
    )
    parser.add_argument("--device", default="cuda", type=str, help="GPU device.")
    parser.add_argument(
        "--max_len",
        default=32,
        type=int,
        help="Maximum length of the sequence after tokenization.",
    )
    parser.add_argument(
        "--dial_path",
        default="../data/dial/sentiment-race/",
        type=str,
        help="DIAL Path.",
    )
    parser.add_argument(
        "--pan16_gender_path",
        default="../data/pan16/gender/",
        type=str,
        help="PAN16 Gender Path.",
    )
    parser.add_argument(
        "--pan16_age_path",
        default="../data/pan16/age/",
        type=str,
        help="PAN16 Age Path.",
    )
    parser.add_argument(
        "--bios_path",
        default="../data/bios/BIOS.pkl",
        type=str,
        help="Biographies corpus path.",
    )
    parser.add_argument(
        "--save_path", default="../data/wizard/", type=str, help="save path."
    )
    return parser


### Utility Functions
def get_sentiment(sentence):
    """
    Returns the sentiment output from VADER given a sentence

    Output: 1 - positive
            0 - neutral
            2 - negative
    """

    sid_obj = SentimentIntensityAnalyzer()
    result = sid_obj.polarity_scores(sentence)

    if result["compound"] >= 0.05:
        return 1
    elif result["compound"] <= -0.05:
        return 0
    else:
        return 2


def load_dump(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


### WIZARD
def load_wizard_(dataset):
    data = []
    for d in dataset:
        if d["gender"] == 2:
            continue

        data.append((d["text"], get_sentiment(d["text"]), d["gender"]))
    return data


def load_wizard():
    wizard = load_dataset("md_gender_bias", "wizard")

    train = load_wizard_(wizard["train"])
    dev = load_wizard_(wizard["validation"])
    test = load_wizard_(wizard["test"])

    return train, dev, test


### FUNPEDIA
def load_funpedia_(dataset):
    data = []
    for d in dataset:
        data.append((d["text"], get_sentiment(d["text"]), d["gender"]))
    return data


def load_funpedia():
    funpedia = load_dataset("md_gender_bias", "funpedia")

    train = load_funpedia_(funpedia["train"])
    dev = load_funpedia_(funpedia["validation"])
    test = load_funpedia_(funpedia["test"])

    return train, dev, test


### CONVAI2
def load_convai_(dataset):
    data = []
    for d in dataset:
        if d["ternary_label"] == 2:
            continue

        data.append((d["text"], get_sentiment(d["text"]), d["ternary_label"]))
    return data


def load_convai():
    wizard = load_dataset("md_gender_bias", "convai2_inferred")

    train = load_convai_(wizard["train"])
    dev = load_convai_(wizard["validation"])
    test = load_convai_(wizard["test"])
    return train, dev, test


### LIGHT


def load_light_(dataset):
    data = []
    for d in dataset:
        if d["ternary_label"] == 2:
            continue

        data.append((d["text"], get_sentiment(d["text"]), d["ternary_label"]))
    return data


def load_light():
    light = load_dataset("md_gender_bias", "light_inferred")

    train = load_light_(light["train"])
    dev = load_light_(light["validation"])
    test = load_light_(light["test"])
    return train, dev, test


### OpenSub


def load_opensub():
    opensub = load_dataset("md_gender_bias", "opensubtitles_inferred")

    train = load_convai_(opensub["train"])
    dev = load_convai_(opensub["validation"])
    test = load_convai_(opensub["test"])
    return train, dev, test


### DIAL


def read_file(path, task, bias):
    content = []
    with open(path, "rb") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            content.append((line.strip(), task, bias))
    return content


def load_dial(path):
    dataset = []
    dataset.extend(read_file(path + "pos_pos", 1, 1))
    dataset.extend(read_file(path + "pos_neg", 1, 0))
    dataset.extend(read_file(path + "neg_pos", 0, 1))
    dataset.extend(read_file(path + "neg_neg", 0, 0))
    return dataset


### BIOS


def load_bios(PATH):
    PROFESSION_MAP = {
        "accountant": 0,
        "architect": 1,
        "attorney": 2,
        "chiropractor": 3,
        "comedian": 4,
        "composer": 5,
        "dentist": 6,
        "dietitian": 7,
        "dj": 8,
        "filmmaker": 9,
        "interior_designer": 10,
        "journalist": 11,
        "model": 12,
        "nurse": 13,
        "painter": 14,
        "paralegal": 15,
        "pastor": 16,
        "personal_trainer": 17,
        "photographer": 18,
        "physician": 19,
        "poet": 20,
        "professor": 21,
        "psychologist": 22,
        "rapper": 23,
        "software_engineer": 24,
        "surgeon": 25,
        "teacher": 26,
        "yoga_teacher": 27,
    }
    GENDER_MAP = {"M": 0, "F": 1}

    dataset = []
    bios = load_dump(PATH)
    for _ in bios:
        dataset.append((_["raw"], PROFESSION_MAP[_["title"]], GENDER_MAP[_["gender"]]))
    shuffle(dataset)
    return dataset


def encode(args, dataset, tokenizer, bert_model, device):
    encoded_dataset = []
    for (sent, label, bias) in tqdm(dataset):
        sent = str(sent)
        sent_emb = torch.tensor(
            [
                tokenizer.encode(
                    sent,
                    max_length=args.max_len,
                    pad_to_max_length=True,
                    truncation=True,
                    add_special_tokens=True,
                )
            ]
        )
        output = bert_model(sent_emb.to(device))[1]
        encoded_dataset.append((output.detach().cpu().numpy(), sent, label, bias))
    return encoded_dataset


def dump_data(filename, dataset):
    with open(filename, "wb") as file:
        pickle.dump(dataset, file)


def get_dataset(args):

    if args.dataset == "wizard":
        train, dev, test = load_wizard()

    if args.dataset == "funpedia":
        train, dev, test = load_funpedia()

    elif args.dataset == "convai2":
        train, dev, test = load_convai()

    elif args.dataset == "light":
        train, dev, test = load_light()

    elif args.dataset == "opensub":
        train, dev, test = load_opensub()

    elif args.dataset == "dial":
        dataset = load_dial(args.dial_path)
        shuffle(dataset)
        dev = []
        train, test = dataset[:166000], dataset[166000:]

    elif args.dataset == "pan16-gender":
        dataset = load_dial(args.pan16_gender_path)
        shuffle(dataset)
        dev = []
        train, test = dataset[:160000], dataset[160000:]

    elif args.dataset == "pan16-age":
        dataset = load_dial(args.pan16_age_path)
        shuffle(dataset)
        dev = []
        train, test = dataset[:160000], dataset[160000:]

    elif args.dataset == "bios":
        data = load_bios(args.bios_path)
        train, dev, test = (
            data[: int(0.65 * len(data))],
            data[int(0.65 * len(data)) : int(0.75 * len(data))],
            data[int(0.75 * len(data)) :],
        )

    return train, dev, test


def save_dataset(args, encoded_data_train, encoded_data_dev, encoded_data_test):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dump_data(os.path.join(args.save_path, "train.pkl"), encoded_data_train)
    dump_data(os.path.join(args.save_path, "dev.pkl"), encoded_data_dev)
    dump_data(os.path.join(args.save_path, "test.pkl"), encoded_data_test)


def main():
    args = get_config().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Tokenizer initialize
    tokenizer = BertTokenizer.from_pretrained(args.MODEL)
    bert_model = BertModel.from_pretrained(args.MODEL)
    bert_model.to(device)

    train, dev, test = get_dataset(args)
    encoded_data_train, encoded_data_dev, encoded_data_test = (
        encode(args, train, tokenizer, bert_model, device),
        encode(args, dev, tokenizer, bert_model, device),
        encode(args, test, tokenizer, bert_model, device),
    )

    save_dataset(args, encoded_data_train, encoded_data_dev, encoded_data_test)


if __name__ == "__main__":
    main()
