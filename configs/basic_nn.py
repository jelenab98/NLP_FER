from models.basic_nn_model import SimpleClassifier
from datetime import datetime

# Model config
glove_dim = 300
hidden_dim = 300
model_constructor = lambda e, f, g: SimpleClassifier(e, embed_dim=glove_dim, features_dim=g, hidden_dim=hidden_dim,
                                                     num_labels=2)

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
save_path = f"../saves/basic_nn_{dt_string}/"

# Test configuration
path_punctuation = "../saves_2/basic_nn_false/best_model.pth"     # path to model with punctuation
path_no_punctuation = "../saves_2/basic_nn_true/basic_nn_10_01_2022_13_47_43/best_model.pth"    # path to model with no punctuation
test_mode = "punctuation"  # can be punctuation or features
exact = False
correction = True
