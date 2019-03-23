import numpy as np
import math as m
import random as r

class NeuralNetwork:

    def __init__(self, init_weights):
        self.weight_mult = 2
        self.layer_num = 1
        self.hidden_num = 20
        self.output_num = 10
        self.input_num = 10
        self.layer_funcs = [self.sigmoid, self.sigmoid]

        self.score = 0.0
        if (init_weights):
            self.init_network()
        else:
            self.W = []

    def init_network(self):
        #Initializes the weights to random values between [0,weight_mult]
        self.W = [np.random.rand(self.input_num, self.hidden_num) * (self.weight_mult*2) - self.weight_mult]
        for i in range(self.layer_num-1):
            self.W.append(np.random.rand(self.hidden_num, self.hidden_num) * (self.weight_mult*2) - self.weight_mult)
        self.W.append(np.random.rand(self.hidden_num, self.output_num) * (self.weight_mult*2) - self.weight_mult)

    def calculate_outputs(self, inputs):
        a = inputs
        for i in range(self.layer_num+1):
            z = np.dot(a.T, self.W[i])
            a = self.layer_funcs[i](z)
        return a

    def sigmoid(self, z):
        sig = lambda x: 1/(1 + m.e**(-x))
        vec_sig = np.vectorize(sig)
        return sig(z).T

    def ReLU(self, z):
        return np.where(z > 0, z, 0)

    def soft_max(self, z):
        exp_sum = 0
        for elem in z[0]:
            exp_sum += m.e**(elem)

        a = np.zeros(z.shape)
        for i in range(len(z[0])):
            a[0][i] = float(m.e**(z[0][i])/exp_sum)
        return a

mutation_percent = 0.1

def genetic_algorithm(networks):
    total_score = 0
    for n in networks:
        total_score += n.score

    #The percentages of each network
    net_perc = [networks[0].score/total_score]
    for i in range(1,len(networks)):
        net_perc.append(net_perc[i-1] + networks[i].score/total_score)

    print(net_perc)

    new_networks = []
    for i in range(len(networks)):
        #Select two networks and "breed" them
        n1 = r.random()
        net1 = -1
        for i in range(len(net_perc)):
            if (n1 <= net_perc[i]):
                net1 = i
                break
        n2 = r.random()
        net2 = -1
        for i in range(len(net_perc)):
            if (n2 <= net_perc[i]):
                net2 = i
                break

        net1 = networks[net1]
        net2 = networks[net2]

        #Now combine the two weights
        new_net = NeuralNetwork(False)
        for i in range(net1.layer_num+1):
            new_w = ((net1.W[i] + net2.W[i])/2.0)
            w_shape = new_w.shape

            #Now apply the mutation
            mut_w = np.random.rand(w_shape[0], w_shape[1])
            ran_w = np.random.rand(w_shape[0], w_shape[1]) * (net1.weight_mult*2) - net1.weight_mult
            new_net.W.append(np.where(mut_w < mutation_percent, ran_w, new_w))

        new_networks.append(new_net)

    return new_networks



# net1 = NeuralNetwork(True)
# net1.score = 5.0
# net2 = NeuralNetwork(True)
# net2.score = 10.0
# net3 = NeuralNetwork(True)
# net3.score = 15.0
# networks = [net1, net2, net3]
# new_networks = genetic_algorithm(networks)
# print(new_networks[0].W)
