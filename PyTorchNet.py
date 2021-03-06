import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import random as r

class NeuralNet(nn.Module):

    def __init__(self, settings):
        super(NeuralNet, self).__init__()
        self.settings = settings

        self.hidden_func = nn.Sigmoid()
        self.output_func = nn.Hardtanh(-100,100)

        self.layers = OrderedDict()
        self.layers[str(0)] = nn.Linear(settings["input_num"], settings["hidden_num"])
        self.layers[str(0) + ' func'] = self.hidden_func
        for i in range(1,settings["layer_num"]+1):
            self.layers[str(i)] = nn.Linear(settings["hidden_num"], settings["hidden_num"])
            self.layers[str(i) + ' func'] = self.hidden_func
        self.layers[str(settings["layer_num"]+1)] = nn.Linear(settings["hidden_num"], settings["output_num"])
        self.layers[str(settings["layer_num"]+1) + ' func'] = self.output_func

        self.inputs = []
        self.batch_size = 80
        self.sample_size = 160
        self.discount = 0.95

        self.model = nn.Sequential(self.layers)

    #Rewards that are gotten at particular time steps and the output choices that were made
    def back_pass(self, rewards, choices):
        real_reward = []
        for i in range(self.batch_size):
            real_reward.append(0)
            for u in range(i, i + self.batch_size):
                real_reward[-1] += rewards[u] * (self.discount**(u-i))

        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.sample_size):
            r_num = r.randint(0,self.batch_size)
            y_pred = self.model(self.inputs[r_num])

            #Just change the one that you know the reward for
            y_actual = y_pred
            y_actual[0][choices[r_num]] = rewards[r_num]

            # Compute and print loss
            loss = criterion(y_pred, y_actual)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.inputs = self.inputs[int(len(self.inputs)/2):]


#This is all just testing the neural nets. I think it works now

# batch_size = 1
# settings = {"layer_num": 5, "hidden_num": 50, "output_num": 5, "input_num": 10}
#
# x = torch.randn(batch_size, settings["input_num"])
# y = torch.tensor([[2, -0.5, 0.23, 0.9, 0.64]])
# print(y)
#
# net = NeuralNet(settings)
# print(net.model.parameters)
#
# #The optimizer for the neural net and the type of loss
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(net.model.parameters(), lr=0.001, momentum=0.9)
# for t in range(500):
#     # Forward pass: Compute predicted y by passing x to the model
#     y_pred = net.model(x)
#
#     # Compute and print loss
#     loss = criterion(y_pred, y)
#     print(t, loss.item())
#
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
