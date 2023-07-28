import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.Linear1 = nn.Linear(
            in_planes, planes)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(planes)
    def forward(self, x):
        out = self.Linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class MLP(nn.Module):

    def __init__(self, in_planes, num_classes = 2):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(MLP, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.Linear = nn.Linear(self.in_planes, 128)
        self.softmax = nn.Softmax(dim=1)
        self.hidden_layers = []
        self.output_layers = []
        self.protected_output_layers = []
        self.layer_numer = 5

        for i in range(self.layer_numer - 1):
            self.hidden_layers.append(BasicBlock(128,128))
        for i in range(self.layer_numer):
            self.output_layers.append(self._make_mlp1(128))
            self.protected_output_layers.append(self._make_mlp1(128))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)  #
        self.output_layers = nn.ModuleList(self.output_layers)  #
        self.protected_output_layers = nn.ModuleList(self.protected_output_layers)  #


    def _make_mlp1(self, in_planes):
        classifier = nn.Sequential(
            nn.Linear(in_planes, self.num_classes),
        )
        return classifier


    def classfier(self, x):
        hidden_connections = []
        hidden_connections.append(F.leaky_relu(self.Linear(x)))
        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))
        output_class = []
        for i in range(len(self.output_layers)):
            output = self.output_layers[i](hidden_connections[i])
            output_class.append(output)
        return output_class

    def adversary(self, x):
        hidden_connections = []
        hidden_connections.append(F.leaky_relu(self.Linear(x)))
        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))
        protect_class = []
        for i in range(len(self.output_layers)):
            protect = self.protected_output_layers[i](hidden_connections[i])
            protect_class.append(protect)
        return protect_class
