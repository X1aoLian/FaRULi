import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.parameter import Parameter

from src.structure import MLP
import logging
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler('./result/logging.txt')])



class Elastic_Tabular:
    def __init__(self, train_loader, train_label, train_protect, columns, theta, theta2, path, lr = 0.0001, b=0.9,
                 eta = -0.001, s=0.008, m=0.99):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.correct_protected = 0
        self.accuracy = 0
        self.lr = lr
        self.path = path
        self.train_loader = train_loader
        self.columns = columns
        self.theta1 = theta
        self.theta2 = theta2
        self.label = train_label
        self.protect = train_protect
        self.number_layer = 5
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.classifier = MLP(self.columns, 2).to(self.device)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), self.lr, weight_decay=1e-5)
        self.Loss_y = nn.CrossEntropyLoss()
        self.Loss_p = nn.CrossEntropyLoss()

        classes = torch.unique(self.label.cpu()).numpy()
        if not np.all(np.isin(np.unique(self.label.cpu().numpy()), classes)):
            raise ValueError("classes should include all valid labels that can be in y")
        class_weights = compute_class_weight(class_weight= 'balanced', classes= classes, y=self.label.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.Loss_y = nn.CrossEntropyLoss(weight= class_weights)

        classes = torch.unique(self.protect.cpu()).numpy()
        if not np.all(np.isin(np.unique(self.protect.cpu().numpy()), classes)):
            raise ValueError("classes should include all valid labels that can be in y")
        class_weights_pro = compute_class_weight(class_weight= 'balanced', classes= classes, y=self.protect.cpu().numpy())
        class_weights_pro = torch.tensor(class_weights_pro, dtype=torch.float32).to(self.device)

        self.Loss_p = nn.CrossEntropyLoss(weight= class_weights_pro)



        self.Accuracy = []
        self.alpha = Parameter(torch.Tensor(self.number_layer).fill_(1 / self.number_layer), requires_grad=False).to(
            self.device)

        self.alpha_loss = Parameter(torch.Tensor(self.number_layer).fill_(0), requires_grad=False)

    def Forward(self):

        for j,(batch) in enumerate(self.train_loader):
            x_batch, y_batch, p_batch = batch
            predictions_per_layer = self.classifier.classfier(x_batch)
            self.optimizer_classifier.zero_grad()
            losses_per_layer = []
            for out in predictions_per_layer:
                loss = self.Loss_y(out, y_batch)
                losses_per_layer.append(loss)
            y_hat = torch.zeros_like(predictions_per_layer[0])
            for i, out in enumerate(predictions_per_layer):
                y_hat += self.alpha[i] * out

            Loss_label_alllayer = self.Loss_y(y_hat, y_batch)
            Loss_label = torch.zeros((self.number_layer,))
            for i, loss in enumerate(losses_per_layer):
                Loss_label[i] = loss
            Loss_label_total = sum(x * y for x, y in zip(Loss_label, self.alpha)) + Loss_label_alllayer
            Loss_label_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer_classifier.step()


            protected_per_layer = self.classifier.classfier(x_batch)
            losses_protected_per_layer = []
            for pro in protected_per_layer:
                loss = self.Loss_p(pro, p_batch)
                losses_protected_per_layer.append(loss)
            p_hat = torch.zeros_like(protected_per_layer[0])
            for i, pro in enumerate(protected_per_layer):
                p_hat += self.alpha[i] * pro

            self.optimizer_classifier.zero_grad()
            Loss_protect = torch.zeros((self.number_layer,))
            Loss_layer = torch.zeros((self.number_layer,))
            for i, loss in enumerate(losses_protected_per_layer):
                Loss_protect[i] = loss
                Loss_layer[i] = Loss_label[i].detach() - self.theta1 * loss

            self.alpha_loss += Loss_layer

            Loss_protect_alllayer = self.Loss_p(p_hat, p_batch)

            Loss_total = sum(x * y for x, y in zip(Loss_layer, self.alpha)) + Loss_label_alllayer.detach() - self.theta1 * Loss_protect_alllayer.detach()

            Loss_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer_classifier.step()

            for i in range(len(losses_per_layer)):
                self.alpha[i] = torch.exp(-(0.1) * self.alpha_loss[i])#?

            z_t = torch.sum(self.alpha)
            self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        #print(f"Total Loss = {(Loss_total.item()):.3f} \
        #                                        Label Loss = {(Loss_label_total.item()):.3f} \
        #                                        Protect Loss = {(Loss_protect_total.item()):.3f}")
        #Q = self.distance(self.alpha)




    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)



    def Test(self, test_data, test_label, p,):
        DR = 0
        DG = 0
        FR = 0
        FG = 0

        y_hat, p_hat,  = self.HB_Test(test_data)


        _, predicted = torch.max(y_hat.data, 1)
        _, predicted_protected = torch.max(p_hat.data, 1)


        F1 = f1_score(test_label.cpu().numpy(), predicted.cpu().numpy())
        label_accuracy = accuracy_score(test_label.cpu().numpy(), predicted.cpu().numpy())
        protect_accuracy = accuracy_score(p.cpu().numpy(), predicted_protected.cpu().numpy())

        for i in range(len(p)):
            if p.cpu()[i] == 0 and predicted.cpu()[i] == 0:
                DR += 1
            elif p.cpu()[i] == 1 and predicted.cpu()[i] == 0:
                FR += 1
            elif p.cpu()[i] == 0 and predicted.cpu()[i] == 1:
                DG += 1
            elif p.cpu()[i] == 1 and predicted.cpu()[i] == 1:
                FG += 1
        logging.info('FG: {}, FR: {}, DG: {}, DR: {}, Discrimation Score: {:.3f}.'
              .format(FG, FR, DG, DR, (FG/(FG+FR)) - (DG)/(DG+DR)))
        #print('FG: {}, FR: {}, DG: {}, DR: {}, Discrimation Score: {:.3f}.'
        #      .format(FG, FR, DG, DR, (FG/(FG+FR)) - (DG)/(DG+DR)))

        logging.info('F1: {:.3f}, Label Acc: {:.3f}, Protected Acc: {:.3f},'
              .format(F1, label_accuracy, protect_accuracy, ))
        #print('F1: {:.3f}, Label Acc: {:.3f}, Protected Acc: {:.3f},'
        #      .format(F1, label_accuracy, protect_accuracy, ))
        print(label_accuracy, (FG/(FG+FR)) - (DG)/(DG+DR))
    def high_confidence_label(self,test_data):
        y_hat, p_hat, = self.HB_Test(test_data)

        _, predicted = torch.max(y_hat.data, 1)
        _, predicted_protected = torch.max(p_hat.data, 1)
        return predicted

    def HB_Test(self, X,):
        predictions_per_layer  = self.classifier.classfier(X)
        protected_per_layer = self.classifier.adversary(X)
        output = torch.zeros_like(predictions_per_layer[0])
        protect = torch.zeros_like(protected_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha[i] * out

        for i, pro in enumerate(protected_per_layer):
            protect += self.alpha[i] * pro

        return output, protect,

    def distance(self, alpha):
        alpha = alpha.cpu().numpy()
        L = len(alpha)
        Q = -sum([l * alpha[l - 1] * math.log(alpha[l - 1]) if alpha[l - 1] > 0 else 0 for l in range(1, L + 1)])
        return Q