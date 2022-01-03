from util.dataloader import PytorchDataset, PytorchFeatureDataset, load_test_data
from util.utils import load_and_preprocess, demo
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import pandas as pd
import numpy as np
import importlib
import torch


if __name__ == '__main__':
    # Define configuration path
    conf_path = Path("../saves/TAR_models/GRU_DNN/punct/config.py")

    # Get configuaration
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    conf.batch_size = 1

    # Set seeds for reproducibility
    seed = conf.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize GloVe
    glove = GloVe(dim=conf.glove_dim)

    # Loading data
    conf.remove_punctuation = False
    conf.use_features = True

    _, _, _, _, x_test, y_test, _, _, _, data_t = load_and_preprocess(conf, padding=True)
    features_test = np.array([feature for feature in data_t]).transpose()
    test_dataset_punctuation = PytorchDataset(x_test, y_test)
    test_dataloader_punctuation = DataLoader(test_dataset_punctuation, batch_size=conf.batch_size, num_workers=2)
    test_dataset_features = PytorchFeatureDataset(x_test, features_test, y_test)
    test_dataloader_punctuation_feat = DataLoader(test_dataset_features, batch_size=conf.batch_size, num_workers=2)

    conf.use_features = False
    conf.remove_punctuation = True
    _, _, _, _, x_test, y_test, _ = load_and_preprocess(conf, padding=True)
    test_dataset = PytorchDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

    df_test = load_test_data("A")
    examples = df_test["text"][:15]
    labels = df_test["label"][:15]

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model without punctuation
    first_model = torch.load(conf.path_punctuation).to(device)
    second_model = torch.load(conf.path_punctuation_features).to(device)
    third_model = torch.load(conf.path_no_punctuation).to(device)

    demo(model_p=first_model, model_no=third_model, model_pf=second_model, data_p=test_dataloader_punctuation,
         data_no=test_dataloader, data_pf=test_dataloader_punctuation_feat,
         inputs=examples, labels=labels, device=device)





