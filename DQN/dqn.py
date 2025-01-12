import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_layer_dim: int = 10):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, action_dim)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = QNet(state_dim=state_dim, action_dim=action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)