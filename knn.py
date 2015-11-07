import csv
import random
import math
import operator
import numpy
from mean_f1 import mean_f1
import matplotlib
import matplotlib.pyplot as plt

 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(len(dataset[0])):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def knn_accuracy(file_name,iteration_no,learning_rate):
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset(file_name, split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	accl = []
	k = 3
	for i in range(iteration_no):
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
			print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		accuracy = getAccuracy(testSet, predictions)
		accl.append(accuracy)
	return accl

def knn_sse(file_name,iteration_no,learning_rate):
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset(file_name, split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	accl = []
	k = 3
	for i in range(iteration_no):
		E=[]
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
			error = float(testSet[x][-1])-float(result)
			E.append(error)
		SSE= numpy.matrix(E)
		SSE= SSE*SSE.T
		SSE= SSE.item()
		accl.append(SSE)
	return accl

def knn_fmeasure(file_name,iteration_no,learning_rate):
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset(file_name, split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	accl = []
	k = 3
	for i in range(iteration_no):
		y_true = []
		y_pred = []
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
			y_true.append([float(testSet[x][-1])])
			y_pred.append([float(result)])
		acc = mean_f1(y_true, y_pred)
		accl.append(acc*100)

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
	return accl
	