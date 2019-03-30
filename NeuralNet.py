import numpy as np
import math as m
import random as r

def sigmoid(z):
    #For some reason thus vectorize thing doesn't always work
    # sig = lambda x: 1/(1 + m.e**(-x))
    # vec_sig = np.vectorize(sig)
    # return vec_sig(z)

    for i in range(len(z)):
        z[i][0] = 1/(1 + m.e**(-z[i][0]))
    return z

def ReLU(z):
    return np.where(z > 0, z, 0)

def soft_max(z):
    exp_sum = 0
    for elem in z[0]:
        exp_sum += m.e**(elem)

    a = np.zeros(z.shape)
    for i in range(len(z[0])):
        a[0][i] = float(m.e**(z[0][i])/exp_sum)
    return a

class NeuralNetwork:

    def __init__(self, init_weights, settings):
        self.weight_mult = settings["weight_mult"]
        self.layer_num = settings["layer_num"]
        self.hidden_num = settings["hidden_num"]
        self.output_num = settings["output_num"]
        self.input_num = settings["input_num"]
        self.layer_funcs = settings["layer_funcs"]

        self.score = 0.0
        if (init_weights):
            self.init_network()
        else:
            self.W = []

        #All of the next stuff is for reinforcement learning
        self.discount_rate = 0.95

        #The rewards at any given time
        self.rewards = []

        #All of the states at a given time (ie the specific inputs)
        self.memory = []

        #The outputs at various timesteps
        self.output_values = []

        #The choice that the network made at a given timestep
        self.out_choice = []

        #The amount of frames before you use backpropagation
        self.memory_length = 100

        #The amount of memories to sample for backpropagation
        self.sample_count = 50

    def init_network(self):
        #Initializes the weights to random values between [0,weight_mult]
        self.W = [np.random.rand(self.input_num, self.hidden_num) * (self.weight_mult*2) - self.weight_mult]
        for i in range(self.layer_num-1):
            self.W.append(np.random.rand(self.hidden_num, self.hidden_num) * (self.weight_mult*2) - self.weight_mult)
        self.W.append(np.random.rand(self.hidden_num, self.output_num) * (self.weight_mult*2) - self.weight_mult)

    def calculate_outputs(self, inputs):
        a = inputs
        z = None
        for i in range(self.layer_num+1):
            z = np.dot(a.T, self.W[i]).T
            a = self.layer_funcs[i](z)
        return self.layer_funcs[self.layer_num](z)

    def calculate_rl_outputs(self, reward, inputs):
        a = inputs
        z = None
        for i in range(self.layer_num+1):
            z = np.dot(a.T, self.W[i]).T
            a = self.layer_funcs[i](z)

        #Now you have to store the inputs, the reward, and the output
        self.memory.append(inputs)
        self.rewards.append(reward)
        self.outputs.append(self.layer_funcs[self.layer_num](z))

        #Check if we should now put the gradient
        if (len(self.memory) == 2*self.memory_length):
            for _ in range(self.sample_count):
                i = r.randint(0,self.memory_length)

                #Select the rewards and inputs for backpropagation
                in = self.memory[i]
                reward = 0
                for u in range(i, i + self.memory_length):
                    reward += self.rewards[u]*self.discount_rate**(u-i)

                #Now calculate the gradients and adjust the weights
                for u in range(len(self.out_choice[i])):
                    out_index = self.out_choice[i][u]
                    expected_reward = self.outputs[out_index][0]
                    calc_rl_gradient(reward, expected_reward, in)

        return self.layer_funcs[self.layer_num](z)

    #UMMMM too lazy to calculate these right now
    def calc_rl_gradient(actual, guess, inputs):
        error = 2 * (actual - guess)
        for i in range(len(self.W)):
            for u in range(i, len(self.W)):


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
