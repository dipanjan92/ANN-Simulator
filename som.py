import matplotlib.pyplot as plt
from itertools import cycle
from random import *
from math import *
from mean_f1 import mean_f1
import numpy

class Node:

    def __init__(self, FV_size=10, PV_size=10, Y=0, X=0):
        self.FV_size=FV_size
        self.PV_size=PV_size
        self.FV = [0.0]*FV_size # Feature Vector
        self.PV = [0.0]*PV_size # Prediction Vector
        self.X=X # X location
        self.Y=Y # Y location
        
        for i in range(FV_size):
            self.FV[i]=random() # Assign a random number from 0 to 1
            
        for i in range(PV_size):
            self.PV[i]=random() # Assign a random number from 0 to 1


class SOM:

    #Let radius=False if you want to autocalculate the radis
    def __init__(self, height=10, width=10, FV_size=10, PV_size=10, radius=False, learning_rate=0.005):
        self.height=height
        self.width=width
        self.radius=radius if radius else (height+width)/2
        self.total=height*width
        self.learning_rate=learning_rate
        self.nodes=[0]*(self.total)
        self.FV_size=FV_size
        self.PV_size=PV_size
        for i in range(self.height):
            for j in range(self.width):
                self.nodes[(i)*(self.width)+j]=Node(FV_size, PV_size,i,j)

    # Train_vector format: [ [FV[0], PV[0]],
    #                        [FV[1], PV[1]], so on..
    
    def train_fmeasure(self, iterations=1000, train_vector=[[[0.0],[0.0]]]):
        y_true=[]
        y_pred=[]
        accl=[]
        time_constant=iterations/log(self.radius)
        radius_decaying=0.0
        learning_rate_decaying=0.0
        influence=0.0
        stack=[] #Stack for storing best matching unit's index and updated FV and PV
        temp_FV=[0.0]*self.FV_size
        temp_PV=[0.0]*self.PV_size
        for i in range(1,iterations+1):
            radius_decaying=self.radius*exp(-1.0*i/time_constant)
            learning_rate_decaying=self.learning_rate*exp(-1.0*i/time_constant)
            
            for  j in range(len(train_vector)):
                input_FV=train_vector[j][0]
                input_PV=train_vector[j][1]
                best=self.best_match(input_FV)
                stack=[]
                for k in range(self.total):
                    dist=self.distance(self.nodes[best],self.nodes[k])
                    if dist < radius_decaying:
                        temp_FV=[0.0]*self.FV_size
                        temp_PV=[0.0]*self.PV_size
                        influence=exp((-1.0*(dist**2))/(2*radius_decaying*i))

                        for l in range(self.FV_size):
                            #Learning
                            temp_FV[l]=self.nodes[k].FV[l]+influence*learning_rate_decaying*(input_FV[l]-self.nodes[k].FV[l])

                        for l in range(self.PV_size):
                            #Learning
                            temp_PV[l]=self.nodes[k].PV[l]+influence*learning_rate_decaying*(input_PV[l]-self.nodes[k].PV[l])

                        #Push the unit onto stack to update in next interval
                        stack[0:0]=[[[k],temp_FV,temp_PV]]

                
                for l in range(len(stack)):
                    
                    self.nodes[stack[l][0][0]].FV[:]=stack[l][1][:]
                    self.nodes[stack[l][0][0]].PV[:]=stack[l][2][:]

                y_true.append(input_PV)
                y_pred.append([int(round(self.predict(input_FV)[0],0))])
            acc = mean_f1(y_true, y_pred)
            accl.append(acc*100)

        return accl

    def train_accuracy(self, iterations=1000, train_vector=[[[0.0],[0.0]]]):
        accl=[]
        time_constant=iterations/log(self.radius)
        radius_decaying=0.0
        learning_rate_decaying=0.0
        influence=0.0
        stack=[] #Stack for storing best matching unit's index and updated FV and PV
        temp_FV=[0.0]*self.FV_size
        temp_PV=[0.0]*self.PV_size
        for i in range(1,iterations+1):
            count = 0
            radius_decaying=self.radius*exp(-1.0*i/time_constant)
            learning_rate_decaying=self.learning_rate*exp(-1.0*i/time_constant)
            
            for  j in range(len(train_vector)):
                input_FV=train_vector[j][0]
                input_PV=train_vector[j][1]
                best=self.best_match(input_FV)
                stack=[]
                for k in range(self.total):
                    dist=self.distance(self.nodes[best],self.nodes[k])
                    if dist < radius_decaying:
                        temp_FV=[0.0]*self.FV_size
                        temp_PV=[0.0]*self.PV_size
                        influence=exp((-1.0*(dist**2))/(2*radius_decaying*i))

                        for l in range(self.FV_size):
                            #Learning
                            temp_FV[l]=self.nodes[k].FV[l]+influence*learning_rate_decaying*(input_FV[l]-self.nodes[k].FV[l])

                        for l in range(self.PV_size):
                            #Learning
                            temp_PV[l]=self.nodes[k].PV[l]+influence*learning_rate_decaying*(input_PV[l]-self.nodes[k].PV[l])

                        #Push the unit onto stack to update in next interval
                        stack[0:0]=[[[k],temp_FV,temp_PV]]

                
                for l in range(len(stack)):
                    
                    self.nodes[stack[l][0][0]].FV[:]=stack[l][1][:]
                    self.nodes[stack[l][0][0]].PV[:]=stack[l][2][:]

                target_val = input_PV[-1]
                predicted_val = int(round(self.predict(input_FV)[0],0))
                if target_val == predicted_val :
                    count += 1
            acc = (count/float(len(train_vector))) * 100.0
            accl.append(acc)

        return accl

    def train_sse(self, iterations=1000, train_vector=[[[0.0],[0.0]]]):
        accl=[]
        time_constant=iterations/log(self.radius)
        radius_decaying=0.0
        learning_rate_decaying=0.0
        influence=0.0
        stack=[] #Stack for storing best matching unit's index and updated FV and PV
        temp_FV=[0.0]*self.FV_size
        temp_PV=[0.0]*self.PV_size
        for i in range(1,iterations+1):
            E = []
            count = 0
            radius_decaying=self.radius*exp(-1.0*i/time_constant)
            learning_rate_decaying=self.learning_rate*exp(-1.0*i/time_constant)
            
            for  j in range(len(train_vector)):
                input_FV=train_vector[j][0]
                input_PV=train_vector[j][1]
                best=self.best_match(input_FV)
                stack=[]
                for k in range(self.total):
                    dist=self.distance(self.nodes[best],self.nodes[k])
                    if dist < radius_decaying:
                        temp_FV=[0.0]*self.FV_size
                        temp_PV=[0.0]*self.PV_size
                        influence=exp((-1.0*(dist**2))/(2*radius_decaying*i))

                        for l in range(self.FV_size):
                            #Learning
                            temp_FV[l]=self.nodes[k].FV[l]+influence*learning_rate_decaying*(input_FV[l]-self.nodes[k].FV[l])

                        for l in range(self.PV_size):
                            #Learning
                            temp_PV[l]=self.nodes[k].PV[l]+influence*learning_rate_decaying*(input_PV[l]-self.nodes[k].PV[l])

                        #Push the unit onto stack to update in next interval
                        stack[0:0]=[[[k],temp_FV,temp_PV]]

                
                for l in range(len(stack)):
                    
                    self.nodes[stack[l][0][0]].FV[:]=stack[l][1][:]
                    self.nodes[stack[l][0][0]].PV[:]=stack[l][2][:]

                target_val = input_PV[-1]
                predicted_val = int(round(self.predict(input_FV)[0],0))
                e = target_val - predicted_val
                E.append(e)
            SSE=numpy.matrix(E)
            SSE=SSE*SSE.T
            SSE=SSE.item()
            accl.append(SSE)
        return accl

                

    #Returns prediction vector
    def predict(self, FV=[0.0]):
        best=self.best_match(FV)
        return self.nodes[best].PV
        
    #Returns best matching unit's index
    def best_match(self, target_FV=[0.0]):

        minimum=sqrt(self.FV_size) #Minimum distance
        minimum_index=1 #Minimum distance unit
        temp=0.0
        for i in range(self.total):
            temp=0.0
            temp=self.FV_distance(self.nodes[i].FV,target_FV)
            if temp<minimum:
                minimum=temp
                minimum_index=i

        return minimum_index

    def FV_distance(self, FV_1=[0.0], FV_2=[0.0]):
        temp=0.0
        for j in range(self.FV_size):
                temp=temp+(FV_1[j]-FV_2[j])**2

        temp=sqrt(temp)
        return temp

    def distance(self, node1, node2):
        return sqrt((node1.X-node2.X)**2+(node1.Y-node2.Y)**2)
    

def som_fmeasure(file_name,iteration_no,learning_rate):
    print("Initialization...")
    a=SOM(5,5,2,1,False,learning_rate)
    k = 3

    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    inputs = []
    dataSet = []
    labels = []
    for p in pat:
        target = []
        input = []
        p=p.split(',')
        t=float(p.pop())
        target.append(t)
        labels.append(int(t))
        for x in p:
            if len(x)>0:
                input.append(float(x))
        inputs.append([input,target])
        dataSet.append(input)

    accl = a.train_fmeasure(iteration_no,inputs)


    if file_name=="iris.csv":
        clusters2 = [[] for _ in range(k)]
        for i, p in enumerate(dataSet):
            clusters2[labels[i]].append(p)
        colors = cycle("rgbcmyk")
        clusters2.sort()
        try:
            from pylab import plot,show,grid,xlabel,ylabel
        except ImportError:
            pass
        else:
            for group, color in zip(clusters2,colors):
                pa,pb,pc,pd = zip(*group)
                x_cords=[]
                y_cords=[]
                for i in range(len(pa)):
                    x_cords.append(pa[i]+pb[i])
                for i in range(len(pc)):
                    y_cords.append(pc[i]+pd[i])

                plot(x_cords,y_cords, "o" + color)
            xlabel('Sepal Length + Sepal Width')
            ylabel('Petal Length + Petal Width')
            grid(True)
            show()

    return accl

def som_sse(file_name,iteration_no,learning_rate):
    print("Initialization...")
    a=SOM(5,5,2,1,False,learning_rate)

    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    inputs = []
    for p in pat:
        target = []
        input = []
        p=p.split(',')
        t=float(p.pop())
        target.append(t)
        for x in p:
            if len(x)>0:
                input.append(float(x))
        inputs.append([input,target])

    print("Training for the dataset...")
    accl = a.train_sse(iteration_no,inputs)

    print("Predictions for the dataset...")

    return accl

def som_accuracy(file_name,iteration_no,learning_rate):
    print("Initialization...")
    a=SOM(5,5,2,1,False,learning_rate)

    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    inputs = []
    for p in pat:
        target = []
        input = []
        p=p.split(',')
        t=float(p.pop())
        target.append(t)
        for x in p:
            if len(x)>0:
                input.append(float(x))
        inputs.append([input,target])

    print("Training for the dataset...")
    accl = a.train_accuracy(iteration_no,inputs)

    print("Predictions for the dataset...")

    return accl