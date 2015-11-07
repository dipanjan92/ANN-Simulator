# Back-Propagation

import math
import random
import string
import numpy,pdb
from mean_f1 import mean_f1
import matplotlib
import matplotlib.pyplot as plt


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)


    def update(self, inputs):

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        error = error + 0.5*(targets-self.ao[k])**2
        return error

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train_sse(self, patterns,iteration_no,learning_rate,momentum_factor):
        # N: learning rate
        SSEL=[]
        for i in range(iteration_no):
            error = 0.0
            E=[]
            for p in patterns:
                p=p.split(',')
                targets=float(p.pop())
                inputs = []
                for x in p:
                    if len(x)>0:
                        inputs.append(float(x))
                self.update(inputs)
                error = error + self.backPropagate(targets, learning_rate, momentum_factor)
                E.append(error)
            SSE=numpy.matrix(E)
            SSE=SSE*SSE.T
            SSE=SSE.item()
            SSEL.append(SSE)
        return SSEL

    def train_accuracy(self, patterns,iteration_no,learning_rate,momentum_factor):
        # N: learning rate
        accl = []
        for i in range(iteration_no):
            error = 0.0
            count = 0
            y_true = []
            y_pred = []
            num=1
            for p in patterns:
                p=p.split(',')
                targets=float(p.pop())
                inputs = []
                for x in p:
                    if len(x)>0:
                        inputs.append(float(x))
                y_true_val = self.update(inputs).pop()
                error = error + self.backPropagate(targets, learning_rate, momentum_factor)
                if abs(round(error,2)) <= 0.01:
                    count+=1
                num = num+1
            acc = (count/float(len(patterns))) * 100.0
            accl.append(acc)
        return accl

    def train_fmeasure(self, patterns,iteration_no,learning_rate,momentum_factor):
        # N: learning rate
        accl = []
        for i in range(iteration_no):
            error = 0.0
            E=[]
            y_true = []
            y_pred = []
            num=1
            for p in patterns:
                p=p.split(',')
                targets=float(p.pop())
                inputs = []
                for x in p:
                    if len(x)>0:
                        inputs.append(float(x))
                y_true_val = self.update(inputs).pop()
                error = error + self.backPropagate(targets, learning_rate, momentum_factor)
                if abs(round(error,2)) <= 0.01:
                    y_true.append([targets])
                else:
                    y_true.append([y_true_val])
                y_pred.append([targets])
                num = num+1
            acc = mean_f1(y_true, y_pred)
            accl.append(acc*100)
        return accl

def backpropagation_sse(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no):
    #opening the patterns
    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    print(len(pat))
    # create a network with two input, two hidden, and one output nodes
    n = NN(input_node_no,hidden_node_no,output_node_no)
    # train it with some patterns
    a = n.train_sse(pat,iteration_no,learning_rate,momentum_factor)
    return a

def backpropagation_accuracy(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no):
    #opening the patterns
    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    print(len(pat))
    # create a network with two input, two hidden, and one output nodes
    n = NN(input_node_no,hidden_node_no,output_node_no)
    # train it with some patterns
    a = n.train_accuracy(pat,iteration_no,learning_rate,momentum_factor)
    return a

def backpropagation_fmeasure(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no):
    #opening the patterns
    pat = []
    if file_name=="iris.csv":
        with open(file_name,'r') as calc:
            pat = calc.read().split('\n')
        print(len(pat))
        x_cord1=[]
        y_cord1=[]
        x_cord2=[]
        y_cord2=[]
        x_cord3=[]
        y_cord3=[]
        for i in pat:
            p = i.split(',')
            targets=float(p.pop())
            inputs = []
            for x in p:
                if len(x)>0:
                    inputs.append(float(x))
            if targets==0.0:
                x_cord1.append(inputs[0]+inputs[1])
                y_cord1.append(inputs[2]+inputs[3])
            if targets==1.0:
                x_cord2.append(inputs[0]+inputs[1])
                y_cord2.append(inputs[2]+inputs[3])
            if targets==2.0:
                x_cord3.append(inputs[0]+inputs[1])
                y_cord3.append(inputs[2]+inputs[3])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        type1 = ax.scatter(x_cord1, y_cord1, s=50, c='red')
        type2 = ax.scatter(x_cord2, y_cord2, s=50, c='green')
        type3 = ax.scatter(x_cord3, y_cord3, s=50, c='blue')
         
        ax.set_title('Petal size vs Sepal size', fontsize=14)
        ax.set_xlabel('Petal size (cm)')
        ax.set_ylabel('Sepal size (cm)')
        ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)
         
        ax.grid(True,linestyle='-',color='0.75')
         
        plt.show()
    # create a network with two input, two hidden, and one output nodes
    n = NN(input_node_no,hidden_node_no,output_node_no)
    # train it with some patterns
    a = n.train_fmeasure(pat,iteration_no,learning_rate,momentum_factor)
    return a