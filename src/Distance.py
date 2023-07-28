import random

import numpy as np
from torch import nn


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



import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler('./result/logging.txt')])



class Elastic_Tabular:
    def __init__(self, train_loader, label, columns, confidence = False, lr = 0.001, b=0.9,
                 eta = -0.001, s=0.008, m=0.99,):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.correct_protected = 0
        self.accuracy = 0
        self.confidence = confidence
        self.lr = lr
        self.train_loader = train_loader
        self.label = label
        self.columns = columns
        self.number_layer = 5
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.classifier = MLP(self.columns, 2).to(self.device)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), self.lr, weight_decay=1e-5)
        self.Loss_d = nn.CrossEntropyLoss()
        self.Loss_p = nn.CrossEntropyLoss()

        classes = torch.unique(self.label.cpu()).numpy()
        if not np.all(np.isin(np.unique(self.label.cpu().numpy()), classes)):
            raise ValueError("classes should include all valid labels that can be in y")
        class_weights = compute_class_weight(class_weight= 'balanced', classes= classes, y=self.label.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.Loss_y = nn.CrossEntropyLoss(weight= class_weights)



        self.Accuracy = []
        self.alpha = Parameter(torch.Tensor(self.number_layer).fill_(1 / self.number_layer), requires_grad=False).to(
            self.device)

        self.alpha_loss = Parameter(torch.Tensor(self.number_layer).fill_(0), requires_grad=False)

    def Forward(self):
        for j,(batch) in enumerate(self.train_loader):
            x_batch, y_batch = batch
            predictions_per_layer = self.classifier.classfier(x_batch)
            self.optimizer_classifier.zero_grad()
            losses_per_layer = []
            for out in predictions_per_layer:
                loss = self.Loss_y(out, y_batch)
                losses_per_layer.append(loss)
            y_hat = torch.zeros_like(predictions_per_layer[0])
            for i, out in enumerate(predictions_per_layer):
                y_hat += self.alpha[i] * out
            _, predicted = torch.max(y_hat.data, 1)

            Loss_label_alllayer = self.Loss_y(y_hat, y_batch)

            Loss_label = torch.zeros((self.number_layer,))
            for i, loss in enumerate(losses_per_layer):
                Loss_label[i] = loss
            Loss_label_total = sum(x * y for x, y in zip(Loss_label, self.alpha)) + Loss_label_alllayer
            Loss_label_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer_classifier.step()

            self.alpha_loss += Loss_label

            for i in range(len(losses_per_layer)):
                self.alpha[i] = torch.exp(-(0.1) * self.alpha_loss[i])#?

            z_t = torch.sum(self.alpha)
            self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        Q = self.distance(self.alpha)
        return Q


    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def distance(self, alpha):
        alpha = alpha.cpu().numpy()
        L = len(alpha)
        Q = -sum([l * alpha[l - 1] * math.log(alpha[l - 1]) if alpha[l - 1] > 0 else 0 for l in range(1, L + 1)])
        return Q

    def Test(self, data, p):
        predictions_per_layer = self.classifier.classfier(data)
        y_hat = torch.zeros_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            y_hat += self.alpha[i] * out
        y_hat = torch.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat.data, 1)
        high_confidence_index = torch.where(y_hat[:, 0] > 0.8)

        high_condifence_d = data[high_confidence_index]

        high_condifence_p = p[high_confidence_index]

        return high_condifence_d, high_condifence_p
def Q_compute(data_1_train, data_2_train, epoch, batch_size):
    random_seed = 1020
    random.seed(random_seed )
    torch.manual_seed(random_seed )
    domain_train = torch.cat((data_1_train, data_2_train))
    domain_train_label = torch.cat((torch.zeros(data_1_train.size(0)), torch.ones(data_2_train.size(0)))).long().to('cuda:0')
    train_data = (domain_train, domain_train_label, )
    train_dataset = TensorDataset(*train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    columns = domain_train.size(1)
    start_1 = Elastic_Tabular(train_loader, domain_train_label,  columns)
    for _ in range(epoch):
        Q  = start_1.Forward()
        torch.cuda.empty_cache()
    return Q

def high_confidence(data_1_train, data_2_train, data_2_protect, epoch = 10, batch_size = 32):
    random_seed = 1020
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    domain_train = torch.cat((data_1_train, data_2_train))
    domain_train_label = torch.cat((torch.zeros(data_1_train.size(0)), torch.ones(data_2_train.size(0)))).long().to(
        'cuda:0')
    train_data = (domain_train, domain_train_label,)
    train_dataset = TensorDataset(*train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True )
    columns = domain_train.size(1)
    start_1 = Elastic_Tabular(train_loader, domain_train_label, columns, )
    for _ in range(epoch):
        Q = start_1.Forward()
    high_condifence_d, high_condifence_p = start_1.Test(data_2_train, data_2_protect)
    return high_condifence_d, high_condifence_p