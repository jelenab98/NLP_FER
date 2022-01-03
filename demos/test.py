import torch

rnn = torch.nn.LSTM(input_size=300, hidden_size=300, dropout=0.2, num_layers=2)