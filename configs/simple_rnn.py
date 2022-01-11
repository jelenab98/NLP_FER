from models.simple_rnn import RNNClassifier
from datetime import datetime

# Model config
glove_dim = 300
hidden_dim = 300
num_layers = 2
dropout = 0.2
model_constructor = lambda e, f, g: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                                  num_layers=num_layers, dropout=dropout, features_dim=g)

# Hyper-parameters
seed = 8008135
batch_size = 32
lr = 1e-4
weight_decay = 5e-4
epochs = 1000
early_stopping = True
early_stop_tolerance = 50

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = False
use_features = True

# Save configuration
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
save_path = f"../saves/simple_rnn_{dt_string}/"

# Test configuration
path_punctuation = "../saves_2/simple_rnn_false/simple_rnn_10_01_2022_15_10_21/best_model.pth"     # path to model with punctuation
path_no_punctuation = "../saves_2/simple_rnn_true/simple_rnn_10_01_2022_15_23_22/best_model.pth"    # path to model with no punctuation
test_mode = "punctuation"  # can be punctuation or features
exact = False
correction = True
