import numpy,random,os,time
from mean_f1 import mean_f1
import matplotlib
import matplotlib.pyplot as plt


def modify_wb(w,learning_rate,t,p):
	w=w+learning_rate*p*t
	return w

def summation(w,p,b):
	net=(w*p.T)+b
	net=net.item()
	return net

def act_func(a):
	if a>0:
		return 1
	if a<0:
		return -1
	if a==0:
		return 0


def perceptron_sse(file_name,iteration_no,learning_rate):
	SSEL=[]
	with open(file_name,'r') as calc:
		l=calc.read().split('\n')
		MCN=0
		x = l[0].split(',')
		w=numpy.random.random(len(x)-1)
		b=random.random()
		w=numpy.matrix(w)
		b=numpy.matrix(b)
		while MCN<iteration_no:
			E=[]
			for i in l:
				j=i.split(',')
				print(j)
				target=float(j.pop())
				p=[]
				for x in j:
					if len(x)>0:
						p.append(float(x))
				p=numpy.matrix(p)
				net=summation(p,w,b)
				a=act_func(net)
				e=target-a
				E.append(e)
				if abs(round(e,2)) >= 0.01:
					w=modify_wb(w,learning_rate,target,p)
			SSE= numpy.matrix(E)
			SSE= SSE*SSE.T
			SSE= SSE.item()
			#SSE=round(SSE,5)
			SSEL.append(SSE)
			MCN = MCN+1
	return SSEL

def perceptron_accuracy(file_name,iteration_no,learning_rate):
	SSEL=[]
	with open(file_name,'r') as calc:
		l=calc.read().split('\n')
		MCN=0
		x = l[0].split(',')
		w=numpy.random.random(len(x)-1)
		b=random.random()
		w=numpy.matrix(w)
		b=numpy.matrix(b)
		while MCN<iteration_no:
			count = 0
			for i in l:
				j=i.split(',')
				target=float(j.pop())
				p=[]
				for x in j:
					if len(x)>0:
						p.append(float(x))
				p=numpy.matrix(p)
				net=summation(p,w,b)
				a=act_func(net)
				e=target-a
				if abs(round(e,2)) <= 0.01:
					count+=1
				else:
					w=modify_wb(w,learning_rate,target,p)

			acc = (count/float(len(l))) * 100.0
			SSEL.append(acc)
			MCN = MCN+1
	return SSEL

def perceptron_fmeasure(file_name,iteration_no,learning_rate):
	SSEL=[]
	with open(file_name,'r') as calc:
		l=calc.read().split('\n')
		MCN=0
		x = l[0].split(',')
		w=numpy.random.random(len(x)-1)
		b=random.random()
		w=numpy.matrix(w)
		b=numpy.matrix(b)
		while MCN<iteration_no:
			y_true = []
			y_pred = []
			for i in l:
				j=i.split(',')
				target=float(j.pop())
				p=[]
				for x in j:
					if len(x)>0:
						p.append(float(x))
				p=numpy.matrix(p)
				net=summation(p,w,b)
				a=act_func(net)
				e=target-a
				if abs(round(e,2)) <= 0.01:
					y_true.append([target])
				else:
					w=modify_wb(w,learning_rate,target,p)
					y_true.append([a])
				y_pred.append([target])

			acc = mean_f1(y_true, y_pred)
			SSEL.append(acc*100)
			MCN = MCN+1
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
	return SSEL