import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model_and_optimizer(input_size, hidden_size, learning_rate):
    model = MLP(input_size, hidden_size).to("cuda:0")
    optimizer = optim.SGD(model.parameters(), learning_rate)
    return model, optimizer


