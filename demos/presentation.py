from util.utils import load_preprocess_one_example, get_predictions, get_lengths
from pathlib import Path

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

    # Set seeds for reproducibility
    seed = conf.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing the best model
    model_no_punct = torch.load(conf.path_no_punctuation)
    model_no_punct.to(device)
    model_no_punct.eval()

    model_punct = torch.load(conf.path_punctuation)
    model_punct.to(device)
    model_punct.eval()

    model_punct_features = torch.load(conf.path_punctuation_features)
    model_punct_features.to(device)
    model_punct_features.eval()

    with torch.no_grad():
        while True:
            example = input("Please enter the input text >> ")
            print(f"Raw input text: {example}\n")
            raw_text = example.strip("\n")

            x_np, x_p, x_pf, f_pf = load_preprocess_one_example(conf, raw_text, device)
            features_pf = torch.tensor(np.array([feature for feature in f_pf]).transpose(), dtype=torch.long).to(device)

            output_no_punct = model_no_punct(x_np, get_lengths(x_np))
            output_punct = model_punct(x_p, get_lengths(x_p))
            output_punct_features = model_punct_features(x_pf, get_lengths(x_pf), features_pf)

            probs_no_punct = torch.sigmoid(output_no_punct)
            probs_punct = torch.sigmoid(output_punct)
            probs_punct_features = torch.sigmoid(output_punct_features)

            class_no_punct = get_predictions(output_no_punct)
            class_punct = get_predictions(output_punct)
            class_punct_features = get_predictions(output_punct_features)

            print(f"Model NoPunct: {100*probs_no_punct.detach().cpu().numpy()[0][class_no_punct]:.2f}% for "
                  f"{'Sarcasm' if class_no_punct == 1 else 'No sarcasm'}")

            print(f"Model Punct: {100*probs_punct.detach().cpu().numpy()[0][class_punct]:.2f}% for "
                  f"{'Sarcasm' if class_punct == 1 else 'No sarcasm'}")

            print(f"Model PunctFeatures: {100*probs_punct_features.detach().cpu().numpy()[0][class_punct_features]:.2f}% for "
                  f"{'Sarcasm' if class_punct_features == 1 else 'No sarcasm'}\n")
