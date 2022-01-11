from tests import mcnemar

from util.dataloader import PytorchDataset, PytorchFeatureDataset
from util.utils import load_and_preprocess
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import torch.nn as nn
import numpy as np
import importlib
import torch

import matplotlib.pyplot as plt
import pandas as pd


def scatter_punct():
    interpunctions = features[:, -1]
    idx_sarcastic = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]
    plt.subplot(1, 2, 1)
    plt.hist(interpunctions[idx_non], color="pink")
    plt.title("Non sarcastic tweets")
    plt.subplot(1, 2, 2)
    plt.hist(interpunctions[idx_sarcastic], color="pink")
    plt.title("Sarcastic tweets")
    #plt.scatter(y_test, interpunctions)
    plt.show()


def check_svm(y_gt_punct, y_gt_no, yt_gt_feat):
    no_punct = pd.read_csv("C:/Users/Jelena/Downloads/no_punct(1).txt", header=None)
    punct = pd.read_csv("C:/Users/Jelena/Downloads/punct(1).txt", header=None)
    punct_features = pd.read_csv("C:/Users/Jelena/Downloads/punct_w_features(1).txt", header=None)
    punct_y = (np.array(punct[0]) == y_gt_punct.squeeze()) * 1
    no_y = (np.array(no_punct[0]) == y_gt_no.squeeze()) * 1
    feat_y = (np.array(punct_features[0]) == yt_gt_feat.squeeze()) * 1
    table_1 = mcnemar.create_contingency_table(punct_y, no_y)
    table_2 = mcnemar.create_contingency_table(punct_y, feat_y)
    mcnemar.make_tests(table_1, exact=True if (table_1[0, 1] + table_1[1, 0] < 25) else False,
                       correction=True, alpha=0.05)
    mcnemar.make_tests(table_2, exact=True if (table_2[0, 1] + table_2[1, 0] < 25) else False,
                       correction=True, alpha=0.05)


if __name__ == '__main__':
    # Define configuration path
    paths = ("basic_nn", "cnn_dnn", "cnn_lstm", "lstm_rnn", "simple_rnn")
    for path in paths:
        conf_path = Path(f"../configs/{path}.py")

        # Get configuaration
        spec = importlib.util.spec_from_file_location('module', conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        print(f"Testing the {path} model for {conf.test_mode} mode")

        # Set seeds for reproducibility
        seed = conf.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # loading and preparing data with punctuation
        conf.remove_punctuation = False
        conf.use_features = False
        x, y, x_val, y_val, x_test, y_test_punct, vocab = load_and_preprocess(conf, padding=True)
        test_dataset_punctuation = PytorchDataset(x_test, y_test_punct)
        test_dataloader_punctuation = DataLoader(test_dataset_punctuation, batch_size=conf.batch_size, num_workers=2)

        # loading and preparing data without punctuation
        conf.use_features = False
        conf.remove_punctuation = True
        x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
        test_dataset = PytorchDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

        # Setting hyper-parameters and models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        first_model = torch.load(conf.path_punctuation)
        first_model = first_model.to(device)
        second_model = torch.load(conf.path_no_punctuation)
        second_model = second_model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Conduct test
        table = mcnemar.evaluate_two_models(first_model, second_model, test_dataloader_punctuation, test_dataloader,
                                            device, criterion, y_test_punct, y_test, num_labels=2, features=False,
                                            batch_size=conf.batch_size)

        mcnemar.make_tests(table, exact=True if (table[0, 1] + table[1, 0] < 25) else False, correction=conf.correction)
