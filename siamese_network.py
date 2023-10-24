import torch
import torch.nn as nn

class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SiameseLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        # Embedding
        x = self.embedding(x)

        # LSTM
        _, (x, _) = self.lstm(x)

        # Flatten LSTM output
        x = x.view(x.size(1), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

