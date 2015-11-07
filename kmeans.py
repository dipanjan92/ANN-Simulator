import sys,math
import matplotlib.pyplot as plt
from itertools import cycle
import numpy
from mean_f1 import mean_f1


def random_picker(dataSet,k):
    from random import choice
    set_c = set(tuple(choice(dataSet)) for _ in range(k))
    i = 0
    n = len(dataSet)
    
    while len(set_c) < k and i < n and i < 4 * k:
        set_c.add( tuple(dataSet[i]))
        i+=1
    list_c = list(map(list,set_c))
    if len(list_c) < k:
        list_c.extend(list_c[[0]] *  (k - len(list_c)))

    return list_c

def euclidianDistance(x1,x2):
    b = (x2[1] - x1[1])**2
    a = (x2[0] - x1[0])**2
    
    return math.sqrt(a + b)

def kmeans(dataSet,k):
    max_distance = 1.797693e308
    n = len(dataSet)
    m = len(dataSet[0])
    
    counts = [0] * k
    labels = [0] * n
    tmp_labels = [0] * n

    arr_c = []
    for i in range(k):
        arr_c.append([0.0] * m)
    list_c = random_picker(dataSet,k)
    mcn = 0
    changed = False
    while True:
        changed = False
        for i in range(k):
            counts[i] = 0
            for j in range(m):
                arr_c[i][j] = 0
        
        
        for h in range(n):
            min_distance  = max_distance
            for i in range(k):
                distance = euclidianDistance(dataSet[h],list_c[i])
                if distance < min_distance:
                    tmp_labels[h] = i
                    min_distance = distance
            
            if tmp_labels[h] != labels[h]:
                changed = True
            
            labels[h] = tmp_labels[h]
            for j in range(m):
                arr_c[labels[h]][j] += dataSet[h][j]
            counts[labels[h]] += 1
        
        for i in range(k):
            for j in range(m):
                list_c[i][j] = arr_c[i][j] / counts[i]  if counts[i] else arr_c[i][j]
            
        
        mcn +=1
        
        if not changed:
            break

    return labels,list_c


def train_sse(dataSet,k,target,iteration_no):
    max_distance = 1.797693e308
    n = len(dataSet)
    m = len(dataSet[0])
    accl=[]
    
    counts = [0] * k
    labels = [0] * n
    tmp_labels = [0] * n

    arr_c = []
    for i in range(k):
        arr_c.append([0.0] * m)
    list_c = random_picker(dataSet,k)
    mcn = 0
    while mcn<iteration_no:
        changed = False
        while True:
            changed = False
            for i in range(k):
                counts[i] = 0
                for j in range(m):
                    arr_c[i][j] = 0
            
            
            for h in range(n):
                E = []
                min_distance  = max_distance
                for i in range(k):
                    distance = euclidianDistance(dataSet[h],list_c[i])
                    if distance < min_distance:
                        tmp_labels[h] = i
                        min_distance = distance
                
                if tmp_labels[h] != labels[h]:
                    changed = True
                
                labels[h] = tmp_labels[h]
                for j in range(m):
                    arr_c[labels[h]][j] += dataSet[h][j]
                counts[labels[h]] += 1

                e = target[h]-labels[h]
                E.append(e)
            SSE= numpy.matrix(E)
            SSE= SSE*SSE.T
            SSE= SSE.item()
            accl.append(SSE)
            
            for i in range(k):
                for j in range(m):
                    list_c[i][j] = arr_c[i][j] / counts[i]  if counts[i] else arr_c[i][j]
                
            
            mcn +=1
            
            if not changed:
                break

    return accl
                
                
def train_accuracy(dataSet,k,target,iteration_no):
    max_distance = 1.797693e308
    n = len(dataSet)
    m = len(dataSet[0])
    accl=[]
    
    counts = [0] * k
    labels = [0] * n
    tmp_labels = [0] * n

    arr_c = []
    for i in range(k):
        arr_c.append([0.0] * m)
    list_c = random_picker(dataSet,k)
    mcn = 0
    while mcn<iteration_no:
        changed = False
        count=0
        while True:
            changed = False
            for i in range(k):
                counts[i] = 0
                for j in range(m):
                    arr_c[i][j] = 0
            
            
            for h in range(n):
                min_distance  = max_distance
                for i in range(k):
                    distance = euclidianDistance(dataSet[h],list_c[i])
                    if distance < min_distance:
                        tmp_labels[h] = i
                        min_distance = distance
                
                if tmp_labels[h] != labels[h]:
                    changed = True
                
                labels[h] = tmp_labels[h]
                for j in range(m):
                    arr_c[labels[h]][j] += dataSet[h][j]
                counts[labels[h]] += 1
                if labels[h]==target[h]:
                    count+=1
            acc = (count/float(n)) * 100.0
            accl.append(acc)
            count = 0
            
            for i in range(k):
                for j in range(m):
                    list_c[i][j] = arr_c[i][j] / counts[i]  if counts[i] else arr_c[i][j]
                
            
            mcn +=1
            
            if not changed:
                break

    return accl                
        
    
    
#Load the dataset
def loadDataset(file_name):
    with open(file_name,'r') as calc:
        pat = calc.read().split('\n')
    inputs = []
    target = []
    for p in pat:
        input = []
        p=p.split(',')
        t = float(p.pop())
        target.append(t)
        for x in p:
            if len(x)>0:
                input.append(float(x))
        inputs.append(input)

    return inputs,target



def kmeans_accuracy(file_name,iteration_no,learning_rate):
    dataSet,target = loadDataset(file_name)
    accl = train_accuracy(dataSet,3,target,iteration_no)
    return accl

def kmeans_sse(file_name,iteration_no,learning_rate):
    dataSet,target = loadDataset(file_name)
    accl = train_sse(dataSet,3,target,iteration_no)
    return accl

def kmeans_fmeasure(file_name,iteration_no,learning_rate):
    dataSet,target = loadDataset(file_name)

    k = 3
    accl =[]

    data_length = len(dataSet)
    for x in range(iteration_no):
        labels, centroids = kmeans(dataSet,k)
        y_true = []
        y_pred = []
        for i in range(data_length):
            y_true.append([target[i]])
            y_pred.append([labels[i]])
        acc = mean_f1(y_true, y_pred)
        accl.append(acc*100)

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
            for group, (ca,cb,cc,cd), color in zip(clusters2,centroids,colors):
                pa,pb,pc,pd = zip(*group)
                x_cords=[]
                y_cords=[]
                for i in range(len(pa)):
                    x_cords.append(pa[i]+pb[i])
                for i in range(len(pc)):
                    y_cords.append(pc[i]+pd[i])

                plot(x_cords,y_cords, "o" + color)

                plot([ca+cb],[cc+cd], '^' + 'y')
            xlabel('Sepal Length + Sepal Width')
            ylabel('Petal Length + Petal Width')
            grid(True)
            show()
    return accl


