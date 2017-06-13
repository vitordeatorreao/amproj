import random as rd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats
import math

def slice (dataset):
	x,y = [],[]
	for i in range(0,len(dataset)):
		temp = dataset[i].split(",")
		y.append(temp.pop(0))
		aux=[]
		for j in range(0,len(temp)):
			aux.append(float(temp[j]))
		x.append(aux)

	return x,y

def extratification(y):

	SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS = [],[],[],[],[],[],[]
	LB = ['SKY', 'CEMENT', 'WINDOW', 'BRICKFACE', 'FOLIAGE', 'PATH','GRASS']

	for i in range(0,len(y)):
		if y[i] == 'SKY':
			SKY.append([y[i],i])
		elif y[i] == 'CEMENT':
			CEMENT.append([y[i],i])
		elif y[i] == 'WINDOW':
			WINDOW.append([y[i],i])
		elif y[i] == 'BRICKFACE':
			BRICKFACE.append([y[i],i])			
		elif y[i] == 'FOLIAGE':
			FOLIAGE.append([y[i],i])
		elif y[i] == 'PATH':
			PATH.append([y[i],i])
		elif y[i] == 'GRASS':
			GRASS.append([y[i],i])

	return SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS

def pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS):
	fold = []
	for i in range(0,30):
		tmp = SKY.pop(rd.randint(0,len(SKY)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = CEMENT.pop(rd.randint(0,len(CEMENT)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = WINDOW.pop(rd.randint(0,len(WINDOW)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = BRICKFACE.pop(rd.randint(0,len(BRICKFACE)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = FOLIAGE.pop(rd.randint(0,len(FOLIAGE)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = PATH.pop(rd.randint(0,len(PATH)-1))
		fold.append(tmp)
	for i in range(0,30):
		tmp = GRASS.pop(rd.randint(0,len(GRASS)-1))
		fold.append(tmp)

	return fold

def makeXFOLD(x,yfold):
	xfold = []
	for i in range (0,len(yfold)):
		xfold.append(x[int(yfold[i][1])])
	return xfold

def makeViewsF(f):
	fold = f[:]
	SHAPe,RGb = [],[]

	for i in range(0,len(fold)):
		aux1=[]
		aux2=[]
		for j in range(0,len(fold[i])):
			if j < 9:
				aux1.append(fold[i][j])
			else: 
				aux2.append(fold[i][j])
		SHAPe.append(aux1)
		RGb.append(aux2)	

	SHAPE = np.array(SHAPe)
	RGB = np.array(RGb)
	return SHAPE,RGB


def kfoldExtratification(x,y):

	SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS = extratification(y)
	
	foldY1 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY2 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY3 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY4 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY5 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY6 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY7 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY8 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY9 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)
	foldY10 = pop30(SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS)

	fold1 = makeXFOLD(x,foldY1)
	fold2 = makeXFOLD(x,foldY2)
	fold3 = makeXFOLD(x,foldY3)
	fold4 = makeXFOLD(x,foldY4)
	fold5 = makeXFOLD(x,foldY5)
	fold6 = makeXFOLD(x,foldY6)
	fold7 = makeXFOLD(x,foldY7)
	fold8 = makeXFOLD(x,foldY8)
	fold9 = makeXFOLD(x,foldY9)
	fold10 = makeXFOLD(x,foldY10)


	train1 = np.array(fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9 + fold10)
	trainy1 = foldY2 + foldY3 + foldY4 + foldY5 + foldY6 + foldY7 + foldY8 + foldY9 + foldY10
	trainY1=[]
	for i in range(0,len(trainy1)):
		trainY1.append(trainy1[i][0])
	test1 = np.array(fold1[:])
	testy1 = foldY1[:]
	testY1=[]
	for i in range(0,len(testy1)):
		testY1.append(testy1[i][0])
	
	train2 = np.array(fold1 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9 + fold10)
	trainy2 = foldY1 + foldY3 + foldY4 + foldY5 + foldY6 + foldY7 + foldY8 + foldY9 + foldY10
	trainY2=[]
	for i in range(0,len(trainy2)):
		trainY2.append(trainy2[i][0])
	test2 = np.array(fold2[:])
	testy2 = foldY2[:]
	testY2=[]
	for i in range(0,len(testy2)):
		testY2.append(testy2[i][0])

	train3 = np.array(fold1 + fold2 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9 + fold10)
	trainy3 = foldY1 + foldY2 + foldY4 + foldY5 + foldY6 + foldY7 + foldY8 + foldY9 + foldY10
	trainY3=[]
	for i in range(0,len(trainy3)):
		trainY3.append(trainy3[i][0])
	test3 = np.array(fold3[:])
	testy3 = foldY3[:]
	testY3=[]
	for i in range(0,len(testy3)):
		testY3.append(testy3[i][0])

	train4 = np.array(fold1 + fold2 + fold3 + fold5 + fold6 + fold7 + fold8 + fold9 + fold10)
	trainy4 = foldY1 + foldY2 + foldY3 + foldY5 + foldY6 + foldY7 + foldY8 + foldY9 + foldY10
	trainY4=[]
	for i in range(0,len(trainy4)):
		trainY4.append(trainy4[i][0])
	test4 = np.array(fold4[:])
	testy4 = foldY4[:]
	testY4=[]
	for i in range(0,len(testy4)):
		testY4.append(testy4[i][0])
		
	train5 = np.array(fold1 + fold2 + fold3 + fold4 + fold6 + fold7 + fold8 + fold9 + fold10)
	trainy5 = foldY1 + foldY2 + foldY3 + foldY4 + foldY6 + foldY7 + foldY8 + foldY9 + foldY10
	trainY5=[]
	for i in range(0,len(trainy5)):
		trainY5.append(trainy5[i][0])
	test5 = np.array(fold5[:])
	testy5 = foldY5[:]
	testY5=[]
	for i in range(0,len(testy5)):
		testY5.append(testy5[i][0])

	train6 = np.array(fold1 + fold2 + fold3 + fold4 + fold5 + fold7 + fold8 + fold9 + fold10)
	trainy6 = foldY1 + foldY2 + foldY3 + foldY4 + foldY5 + foldY7 + foldY8 + foldY9 + foldY10
	trainY6=[]
	for i in range(0,len(trainy6)):
		trainY6.append(trainy6[i][0])
	test6 = np.array(fold6[:])
	testy6 = foldY6[:]
	testY6=[]
	for i in range(0,len(testy6)):
		testY6.append(testy6[i][0])

	train7 = np.array(fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold8 + fold9 + fold10)
	trainy7 = foldY1 + foldY2 + foldY3 + foldY4 + foldY5 + foldY6 + foldY8 + foldY9 + foldY10
	trainY7=[]
	for i in range(0,len(trainy7)):
		trainY7.append(trainy7[i][0])
	test7 = np.array(fold7[:])
	testy7 = foldY7[:]
	testY7=[]
	for i in range(0,len(testy7)):
		testY7.append(testy7[i][0])

	train8 = np.array(fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold9 + fold10)
	trainy8 = foldY1 + foldY2 + foldY3 + foldY4 + foldY5 + foldY6 + foldY7 + foldY9 + foldY10
	trainY8=[]
	for i in range(0,len(trainy8)):
		trainY8.append(trainy8[i][0])
	test8 = np.array(fold8[:])
	testy8 = foldY8[:]
	testY8=[]
	for i in range(0,len(testy8)):
		testY8.append(testy8[i][0])

	train9 = np.array(fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8 + fold10)
	trainy9 = foldY1 + foldY2 + foldY3 + foldY4 + foldY5 + foldY6 + foldY7 + foldY8 + foldY10
	trainY9=[]
	for i in range(0,len(trainy9)):
		trainY9.append(trainy9[i][0])
	test9 = np.array(fold9[:])
	testy9 = foldY9[:]
	testY9=[]
	for i in range(0,len(testy9)):
		testY9.append(testy9[i][0])

	train10 = np.array(fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9)
	trainy10 = foldY1 + foldY2 + foldY3 + foldY4 + foldY5 + foldY6 + foldY7 + foldY8 + foldY9
	trainY10=[]
	for i in range(0,len(trainy10)):
		trainY10.append(trainy10[i][0])
	test10 = np.array(fold10[:])
	testy10 = foldY10[:]
	testY10=[]
	for i in range(0,len(testy10)):
		testY10.append(testy10[i][0])
	

	return 	train1,trainY1,test1,testY1,train2,trainY2,test2,testY2,train3,trainY3,test3,testY3,train4,trainY4,test4,testY4,train5,trainY5,test5,testY5,train6,trainY6,test6,testY6,train7,trainY7,test7,testY7,train8,trainY8,test8,testY8,train9,trainY9,test9,testY9,train10,trainY10,test10,testY10

def getIc(sample):

	z_critical = stats.norm.ppf(q = 0.975)
	pop_std = np.std(sample)
	margin_of_error = z_critical * (pop_std/math.sqrt(len(sample)))
	sample_mean = np.mean(sample)
	confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
	print("Confidence interval:")
	print(confidence_interval)
	return confidence_interval

def errorRate(pred,true):
	count=0
	for i in range(0,len(true)):
		if pred[i]!=true[i]:
			count+=1

	return (count*100)/len(true)

def votoMajoritario(knnShape,knnView,bayesShape,bayesView):
	clfResult= []
	for i in range(0,len(knnShape)):
		aux=[]
		aux.append(knnShape[i])
		aux.append(knnView[i])
		aux.append(bayesShape[i])
		aux.append(bayesView[i])
		cnt = Counter(aux)
		Num = dict(cnt)
		clfResult.append(max(Num, key=Num.get))

	return clfResult



#KNN
def getKNearest(Train,Test,K):
	KNN = []
	for i in range(0,len(Test)):
		dissimilarity= []
		for j in range(0,len(Train)):
			dissimilarity.append(np.sqrt(np.sum((Train[j]-Test[i])**2)))	#EuclideanDistance
		Dissimilarity = np.array(dissimilarity)							#Distance from the sample to all trainSet
		indexes =[]
		for i in range(0,K):
			indexes.append(Dissimilarity.argmin())
			Dissimilarity[Dissimilarity.argmin()] += 1000
		KNN.append(indexes)
			
	return KNN

def makeClassification(trvY,KNN):
	clfResult = []
	for i in range(0,len(KNN)):
		tempLabels = []
		for j in range(0,len(KNN[i])):
			tempLabels.append(trvY[KNN[i][j]])
		cnt = Counter(tempLabels)
		Num = dict(cnt)
		clfResult.append(max(Num, key=Num.get))
	return clfResult





#BAYES

def meanStdClass(x,y):

	SKY,CEMENT,WINDOW,BRICKFACE,FOLIAGE,PATH,GRASS = [],[],[],[],[],[],[]
	LB = ['SKY', 'CEMENT', 'WINDOW', 'BRICKFACE', 'FOLIAGE', 'PATH','GRASS']

	for i in range(0,len(y)):
		if y[i][0] == 'SKY':
			SKY.append(x[i])
		elif y[i][0] == 'CEMENT':
			CEMENT.append(x[i])
		elif y[i][0] == 'WINDOW':
			WINDOW.append(x[i])
		elif y[i][0] == 'BRICKFACE':
			BRICKFACE.append(x[i])			
		elif y[i][0] == 'FOLIAGE':
			FOLIAGE.append(x[i])
		elif y[i][0] == 'PATH':
			PATH.append(x[i])
		elif y[i][0] == 'GRASS':
			GRASS.append(x[i])


	return np.mean(SKY,axis=0),np.std(SKY,axis=0)



