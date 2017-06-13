import util
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from sklearn.naive_bayes import GaussianNB


arq = open('segmentationTest.txt','r')

dataset = []

for line in arq:
    dataset.append(line)

arq.close()

x,y= util.slice(dataset)

viewShapeKnnResult30F,viewRGBKnnResult30F,viewShapeBayesResult30F,viewRGbBayesResult30F,VMJTResult30F = [],[],[],[],[]

for i in range(30):
	train1,trainY1,test1,testY1,train2,trainY2,test2,testY2,train3,trainY3,test3,testY3,train4,trainY4,test4,testY4,train5,trainY5,test5,testY5,train6,trainY6,test6,testY6,train7,trainY7,test7,testY7,train8,trainY8,test8,testY8,train9,trainY9,test9,testY9,train10,trainY10,test10,testY10 = util.kfoldExtratification(x,y)
	viewShapeKnnResult,viewRGBKnnResult,viewShapeBayesResult,viewRGbBayesResult,VMJTResult = [],[],[],[],[]

	#Fold1
	shapeTr1,rgbTr1=util.makeViewsF(train1)
	shapeTst1,rgbTst1=util.makeViewsF(test1)
	
	#KNN
	
	#view1
	KNN1v1 = util.getKNearest(shapeTr1,shapeTst1,15)
	KNN1v1result = util.makeClassification(trainY1,KNN1v1)
	KNN1v1Error = util.errorRate(KNN1v1result,testY1)
	viewShapeKnnResult.append(KNN1v1Error)

	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 1:\n")
	for k in KNN1v1result:
	    file.write(str(k) + '\n') 
	file.close()

	#view2
	KNN1v2 = util.getKNearest(rgbTr1,rgbTst1,15)
	KNN1v2result = util.makeClassification(trainY1,KNN1v2)
	KNN1v2Error = util.errorRate(KNN1v2result,testY1)
	viewRGBKnnResult.append(KNN1v2Error)
	
	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 1:\n")
	for k in KNN1v2result:
	    file.write(str(k) + '\n') 
	file.close()

	#Bayes
	Bayes = GaussianNB()

	Bayes1v1result = Bayes.fit(shapeTr1, trainY1).predict(shapeTst1)
	Bayes1v1Error = util.errorRate(Bayes1v1result,testY1)
	viewShapeBayesResult.append(Bayes1v1Error)

	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 1:\n")
	for b in Bayes1v1result:
	    file.write(str(b) + '\n') 
	file.close()

	Bayes1v2result = Bayes.fit(rgbTr1, trainY1).predict(rgbTst1)
	Bayes1v2Error = util.errorRate(Bayes1v2result,testY1)
	viewRGbBayesResult.append(Bayes1v2Error)

	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 1:\n")
	for b in Bayes1v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VotoMajoritario
	
	VMJT1Result= util.votoMajoritario(KNN1v1result,KNN1v2result,Bayes1v1result,Bayes1v2result)
	VMJT1Error= util.errorRate(VMJT1Result,testY1)
	VMJTResult.append(VMJT1Error)

	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 1:\n")
	for v in VMJT1Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold2
	shapeTr2,rgbTr2=util.makeViewsF(train2)
	shapeTst2,rgbTst2=util.makeViewsF(test2)
	
	#KNN
	
	#view1
	KNN2v1 = util.getKNearest(shapeTr2,shapeTst2,15)
	KNN2v1result = util.makeClassification(trainY2,KNN2v1)
	KNN2v1Error = util.errorRate(KNN2v1result,testY2)
	viewShapeKnnResult.append(KNN2v1Error)

	#view2
	KNN2v2 = util.getKNearest(rgbTr2,rgbTst2,15)
	KNN2v2result = util.makeClassification(trainY2,KNN2v2)
	KNN2v2Error = util.errorRate(KNN2v2result,testY2)
	viewRGBKnnResult.append(KNN2v2Error)

	#Bayes
	Bayes = GaussianNB()
	#view1
	Bayes2v1result = Bayes.fit(shapeTr2, trainY2).predict(shapeTst2)
	Bayes2v1Error = util.errorRate(Bayes2v1result,testY2)
	viewShapeBayesResult.append(Bayes2v1Error)

	#view2
	Bayes2v2result = Bayes.fit(rgbTr2, trainY2).predict(rgbTst2)
	Bayes2v2Error = util.errorRate(Bayes2v2result,testY2)
	viewRGbBayesResult.append(Bayes2v2Error)
	
	#VotoMajoritario
	
	VMJT2Result= util.votoMajoritario(KNN2v1result,KNN2v2result,Bayes2v1result,Bayes2v2result)
	VMJT2Error= util.errorRate(VMJT2Result,testY2)
	VMJTResult.append(VMJT2Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 2:\n")
	for k in KNN2v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 2:\n")
	for k in KNN2v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 2:\n")
	for b in Bayes2v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 2:\n")
	for b in Bayes2v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 2:\n")
	for v in VMJT2Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold3
	shapeTr3,rgbTr3=util.makeViewsF(train3)
	shapeTst3,rgbTst3=util.makeViewsF(test3)
	
	#KNN
	
	#view1
	KNN3v1 = util.getKNearest(shapeTr3,shapeTst3,15)
	KNN3v1result = util.makeClassification(trainY3,KNN3v1)
	KNN3v1Error = util.errorRate(KNN3v1result,testY3)
	viewShapeKnnResult.append(KNN3v1Error)

	#view2
	KNN3v2 = util.getKNearest(rgbTr3,rgbTst3,15)
	KNN3v2result = util.makeClassification(trainY3,KNN3v2)
	KNN3v2Error = util.errorRate(KNN3v2result,testY3)
	viewRGBKnnResult.append(KNN3v2Error)

	#Bayes
	Bayes = GaussianNB()
	#view1
	Bayes3v1result = Bayes.fit(shapeTr3, trainY3).predict(shapeTst3)
	Bayes3v1Error = util.errorRate(Bayes3v1result,testY3)
	viewShapeBayesResult.append(Bayes3v1Error)

	#view2
	Bayes3v2result = Bayes.fit(rgbTr3, trainY3).predict(rgbTst3)
	Bayes3v2Error = util.errorRate(Bayes3v2result,testY3)
	viewRGbBayesResult.append(Bayes3v2Error)
	
	#VotoMajoritario
	
	VMJT3Result= util.votoMajoritario(KNN3v1result,KNN3v2result,Bayes3v1result,Bayes3v2result)
	VMJT3Error= util.errorRate(VMJT3Result,testY3)
	VMJTResult.append(VMJT3Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 3:\n")
	for k in KNN3v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 3:\n")
	for k in KNN3v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 3:\n")
	for b in Bayes3v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 3:\n")
	for b in Bayes3v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 3:\n")
	for v in VMJT3Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold4
	
	shapeTr4,rgbTr4=util.makeViewsF(train4)
	shapeTst4,rgbTst4=util.makeViewsF(test4)
	
	#KNN
	
	#view1
	KNN4v1 = util.getKNearest(shapeTr4,shapeTst4,15)
	KNN4v1result = util.makeClassification(trainY4,KNN4v1)
	KNN4v1Error = util.errorRate(KNN4v1result,testY4)
	viewShapeKnnResult.append(KNN4v1Error)

	#view2
	KNN4v2 = util.getKNearest(rgbTr4,rgbTst4,15)
	KNN4v2result = util.makeClassification(trainY4,KNN4v2)
	KNN4v2Error = util.errorRate(KNN4v2result,testY4)
	viewRGBKnnResult.append(KNN4v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes4v1result = Bayes.fit(shapeTr4, trainY4).predict(shapeTst4)
	Bayes4v1Error = util.errorRate(Bayes4v1result,testY4)
	viewShapeBayesResult.append(Bayes4v1Error)

	#view2
	Bayes4v2result = Bayes.fit(rgbTr4, trainY4).predict(rgbTst4)
	Bayes4v2Error = util.errorRate(Bayes4v2result,testY4)
	viewRGbBayesResult.append(Bayes4v2Error)
	
	#VotoMajoritario
	
	VMJT4Result= util.votoMajoritario(KNN4v1result,KNN4v2result,Bayes4v1result,Bayes4v2result)
	VMJT4Error= util.errorRate(VMJT4Result,testY4)
	VMJTResult.append(VMJT4Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 4:\n")
	for k in KNN4v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 4:\n")
	for k in KNN4v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 4:\n")
	for b in Bayes4v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 4:\n")
	for b in Bayes4v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 4:\n")
	for v in VMJT4Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold5
	
	shapeTr5,rgbTr5=util.makeViewsF(train5)
	shapeTst5,rgbTst5=util.makeViewsF(test5)
	
	#KNN
	
	#view1
	KNN5v1 = util.getKNearest(shapeTr5,shapeTst5,15)
	KNN5v1result = util.makeClassification(trainY5,KNN5v1)
	KNN5v1Error = util.errorRate(KNN5v1result,testY5)
	viewShapeKnnResult.append(KNN5v1Error)

	#view2
	KNN5v2 = util.getKNearest(rgbTr5,rgbTst5,15)
	KNN5v2result = util.makeClassification(trainY5,KNN5v2)
	KNN5v2Error = util.errorRate(KNN5v2result,testY5)
	viewRGBKnnResult.append(KNN5v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes5v1result = Bayes.fit(shapeTr5, trainY5).predict(shapeTst5)
	Bayes5v1Error = util.errorRate(Bayes5v1result,testY5)
	viewShapeBayesResult.append(Bayes5v1Error)

	#view2
	Bayes5v2result = Bayes.fit(rgbTr5, trainY5).predict(rgbTst5)
	Bayes5v2Error = util.errorRate(Bayes5v2result,testY5)
	viewRGbBayesResult.append(Bayes5v2Error)
	
	#VotoMajoritario
	
	VMJT5Result= util.votoMajoritario(KNN5v1result,KNN5v2result,Bayes5v1result,Bayes5v2result)
	VMJT5Error= util.errorRate(VMJT5Result,testY5)
	VMJTResult.append(VMJT5Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 5:\n")
	for k in KNN5v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 5:\n")
	for k in KNN5v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 5:\n")
	for b in Bayes5v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 5:\n")
	for b in Bayes5v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 5:\n")
	for v in VMJT5Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold6
	
	shapeTr6,rgbTr6=util.makeViewsF(train6)
	shapeTst6,rgbTst6=util.makeViewsF(test6)
	
	#KNN
	
	#view1
	KNN6v1 = util.getKNearest(shapeTr6,shapeTst6,15)
	KNN6v1result = util.makeClassification(trainY6,KNN6v1)
	KNN6v1Error = util.errorRate(KNN6v1result,testY6)
	viewShapeKnnResult.append(KNN6v1Error)

	#view2
	KNN6v2 = util.getKNearest(rgbTr6,rgbTst6,15)
	KNN6v2result = util.makeClassification(trainY6,KNN6v2)
	KNN6v2Error = util.errorRate(KNN6v2result,testY6)
	viewRGBKnnResult.append(KNN6v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes6v1result = Bayes.fit(shapeTr6, trainY6).predict(shapeTst6)
	Bayes6v1Error = util.errorRate(Bayes6v1result,testY6)
	viewShapeBayesResult.append(Bayes6v1Error)

	#view2
	Bayes6v2result = Bayes.fit(rgbTr6, trainY6).predict(rgbTst6)
	Bayes6v2Error = util.errorRate(Bayes6v2result,testY6)
	viewRGbBayesResult.append(Bayes6v2Error)
	
	#VotoMajoritario
	
	VMJT6Result= util.votoMajoritario(KNN6v1result,KNN6v2result,Bayes6v1result,Bayes6v2result)
	VMJT6Error= util.errorRate(VMJT6Result,testY6)
	VMJTResult.append(VMJT6Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 6:\n")
	for k in KNN6v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 6:\n")
	for k in KNN6v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 6:\n")
	for b in Bayes6v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 6:\n")
	for b in Bayes6v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 6:\n")
	for v in VMJT6Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold7
	
	shapeTr7,rgbTr7=util.makeViewsF(train7)
	shapeTst7,rgbTst7=util.makeViewsF(test7)
	
	#KNN
	
	#view1
	KNN7v1 = util.getKNearest(shapeTr7,shapeTst7,15)
	KNN7v1result = util.makeClassification(trainY7,KNN7v1)
	KNN7v1Error = util.errorRate(KNN7v1result,testY7)
	viewShapeKnnResult.append(KNN7v1Error)

	#view2
	KNN7v2 = util.getKNearest(rgbTr7,rgbTst7,15)
	KNN7v2result = util.makeClassification(trainY7,KNN7v2)
	KNN7v2Error = util.errorRate(KNN7v2result,testY7)
	viewRGBKnnResult.append(KNN7v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes7v1result = Bayes.fit(shapeTr7, trainY7).predict(shapeTst7)
	Bayes7v1Error = util.errorRate(Bayes7v1result,testY7)
	viewShapeBayesResult.append(Bayes7v1Error)

	#view2
	Bayes7v2result = Bayes.fit(rgbTr7, trainY7).predict(rgbTst7)
	Bayes7v2Error = util.errorRate(Bayes7v2result,testY7)
	viewRGbBayesResult.append(Bayes7v2Error)
	
	#VotoMajoritario
	
	VMJT7Result= util.votoMajoritario(KNN7v1result,KNN7v2result,Bayes7v1result,Bayes7v2result)
	VMJT7Error= util.errorRate(VMJT7Result,testY7)
	VMJTResult.append(VMJT7Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 7:\n")
	for k in KNN7v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 7:\n")
	for k in KNN7v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 7:\n")
	for b in Bayes7v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 7:\n")
	for b in Bayes7v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 7:\n")
	for v in VMJT7Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold8
	
	shapeTr8,rgbTr8=util.makeViewsF(train8)
	shapeTst8,rgbTst8=util.makeViewsF(test8)
	
	#KNN
	
	#view1
	KNN8v1 = util.getKNearest(shapeTr8,shapeTst8,15)
	KNN8v1result = util.makeClassification(trainY8,KNN8v1)
	KNN8v1Error = util.errorRate(KNN8v1result,testY8)
	viewShapeKnnResult.append(KNN8v1Error)

	#view2
	KNN8v2 = util.getKNearest(rgbTr8,rgbTst8,15)
	KNN8v2result = util.makeClassification(trainY8,KNN8v2)
	KNN8v2Error = util.errorRate(KNN8v2result,testY8)
	viewRGBKnnResult.append(KNN8v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes8v1result = Bayes.fit(shapeTr8, trainY8).predict(shapeTst8)
	Bayes8v1Error = util.errorRate(Bayes8v1result,testY8)
	viewShapeBayesResult.append(Bayes8v1Error)

	#view2
	Bayes8v2result = Bayes.fit(rgbTr8, trainY8).predict(rgbTst8)
	Bayes8v2Error = util.errorRate(Bayes8v2result,testY8)
	viewRGbBayesResult.append(Bayes8v2Error)
	
	#VotoMajoritario
	
	VMJT8Result= util.votoMajoritario(KNN8v1result,KNN8v2result,Bayes8v1result,Bayes8v2result)
	VMJT8Error= util.errorRate(VMJT8Result,testY8)
	VMJTResult.append(VMJT8Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 8:\n")
	for k in KNN8v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 8:\n")
	for k in KNN8v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 8:\n")
	for b in Bayes8v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 8:\n")
	for b in Bayes8v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 8:\n")
	for v in VMJT8Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold9
	
	shapeTr9,rgbTr9=util.makeViewsF(train9)
	shapeTst9,rgbTst9=util.makeViewsF(test9)
	
	#KNN
	
	#view1
	KNN9v1 = util.getKNearest(shapeTr9,shapeTst9,15)
	KNN9v1result = util.makeClassification(trainY9,KNN9v1)
	KNN9v1Error = util.errorRate(KNN9v1result,testY9)
	viewShapeKnnResult.append(KNN9v1Error)

	#view2
	KNN9v2 = util.getKNearest(rgbTr9,rgbTst9,15)
	KNN9v2result = util.makeClassification(trainY9,KNN9v2)
	KNN9v2Error = util.errorRate(KNN9v2result,testY9)
	viewRGBKnnResult.append(KNN9v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes9v1result = Bayes.fit(shapeTr9, trainY9).predict(shapeTst9)
	Bayes9v1Error = util.errorRate(Bayes9v1result,testY9)
	viewShapeBayesResult.append(Bayes9v1Error)

	#view2
	Bayes9v2result = Bayes.fit(rgbTr9, trainY9).predict(rgbTst9)
	Bayes9v2Error = util.errorRate(Bayes9v2result,testY9)
	viewRGbBayesResult.append(Bayes9v2Error)
	
	#VotoMajoritario
	
	VMJT9Result= util.votoMajoritario(KNN9v1result,KNN9v2result,Bayes9v1result,Bayes9v2result)
	VMJT9Error= util.errorRate(VMJT9Result,testY9)
	VMJTResult.append(VMJT9Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 9:\n")
	for k in KNN9v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 9:\n")
	for k in KNN9v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 9:\n")
	for b in Bayes9v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 9:\n")
	for b in Bayes9v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 9:\n")
	for v in VMJT9Result:
	    file.write(str(v) + '\n') 
	file.close()

	#Fold10
	
	shapeTr10,rgbTr10=util.makeViewsF(train10)
	shapeTst10,rgbTst10=util.makeViewsF(test10)
	
	#KNN
	
	#view1
	KNN10v1 = util.getKNearest(shapeTr10,shapeTst10,15)
	KNN10v1result = util.makeClassification(trainY10,KNN10v1)
	KNN10v1Error = util.errorRate(KNN10v1result,testY10)
	viewShapeKnnResult.append(KNN10v1Error)

	#view2
	KNN10v2 = util.getKNearest(rgbTr10,rgbTst10,15)
	KNN10v2result = util.makeClassification(trainY10,KNN10v2)
	KNN10v2Error = util.errorRate(KNN10v2result,testY10)
	viewRGBKnnResult.append(KNN10v2Error)

	#Bayes
	Bayes = GaussianNB()

	#view1
	Bayes10v1result = Bayes.fit(shapeTr10, trainY10).predict(shapeTst10)
	Bayes10v1Error = util.errorRate(Bayes10v1result,testY10)
	viewShapeBayesResult.append(Bayes10v1Error)

	#view2
	Bayes10v2result = Bayes.fit(rgbTr10, trainY10).predict(rgbTst10)
	Bayes10v2Error = util.errorRate(Bayes10v2result,testY10)
	viewRGbBayesResult.append(Bayes10v2Error)
	
	#VotoMajoritario
	
	VMJT10Result= util.votoMajoritario(KNN10v1result,KNN10v2result,Bayes10v1result,Bayes10v2result)
	VMJT10Error= util.errorRate(VMJT10Result,testY10)
	VMJTResult.append(VMJT10Error)

	#KNN
	file = open('KNN1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 10:\n")
	for k in KNN10v1result:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('KNN2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 10:\n")
	for k in KNN10v2result:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('BAYES1.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 10:\n")
	for b in Bayes10v1result:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('BAYES2.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 10:\n")
	for b in Bayes10v2result:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VTMJ.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	file.write("fold 10:\n")
	for v in VMJT10Result:
	    file.write(str(v) + '\n') 
	file.close()


	print "Time: " + str(i)

	print "Kfold Result: "

	print 'viewShapeKnnResult: '+ str(np.mean(viewShapeKnnResult))
	print 'viewRGBKnnResult: '+ str(np.mean(viewRGBKnnResult))
	print 'viewShapeBayesResult: '+ str(np.mean(viewShapeBayesResult))
	print 'viewRGbBayesResult: '+ str(np.mean(viewRGbBayesResult))
	print 'VMJTResult: '+ str(np.mean(VMJTResult))

	#KNN
	file = open('viewShapeKnnResult.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	for k in viewShapeKnnResult:
	    file.write(str(k) + '\n') 
	file.close()

	file = open('viewRGBKnnResult.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	for k in viewRGBKnnResult:
	    file.write(str(k) + '\n') 
	file.close()
	
	#BAYES
	file = open('viewShapeBayesResult.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	for b in viewShapeBayesResult:
	    file.write(str(b) + '\n') 
	file.close()
	
	file = open('viewRGbBayesResult.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	for b in viewRGbBayesResult:
	    file.write(str(b) + '\n') 
	file.close()

	#VTMJ
	file = open('VMJTResult.txt','a')

	file.write("\n")
	file.write("time "+str(i)+':\n')
	file.write("\n")
	for v in VMJTResult :
	    file.write(str(v) + '\n') 
	file.close()

	viewShapeKnnResult30F.append(np.mean(viewShapeKnnResult))
	viewRGBKnnResult30F.append(np.mean(viewRGBKnnResult))
	viewShapeBayesResult30F.append(np.mean(viewShapeBayesResult))
	viewRGbBayesResult30F.append(np.mean(viewRGbBayesResult))
	VMJTResult30F.append(np.mean(VMJTResult))


print "Estimativa Pontual para Media:"

print 'viewShapeKnnResult: '+ str(np.mean(viewShapeKnnResult30F))
print 'viewRGBKnnResult: '+ str(np.mean(viewRGBKnnResult30F))
print 'viewShapeBayesResult: '+ str(np.mean(viewShapeBayesResult30F))
print 'viewRGbBayesResult: '+ str(np.mean(viewRGbBayesResult30F))
print 'CombinadosVotoMajoritarioResult: '+ str(np.mean(VMJTResult30F))

print "Desvio Padrao:"

print 'viewShapeKnnResult: '+ str(np.std(viewShapeKnnResult30F))
print 'viewRGBKnnResult: '+ str(np.std(viewRGBKnnResult30F))
print 'viewShapeBayesResult: '+ str(np.std(viewShapeBayesResult30F))
print 'viewRGbBayesResult: '+ str(np.std(viewRGbBayesResult30F))
print 'CombinadosVotoMajoritarioResult: '+ str(np.std(VMJTResult30F))

print 'Estimativas Intervaladas com Confianca com 95%% para a media:'
print 'viewShapeKnnResult: '
util.getIc(viewShapeKnnResult30F)
print 'viewRGBKnnResult: '
util.getIc(viewRGBKnnResult30F)
print 'viewShapeBayesResult: '
util.getIc(viewShapeBayesResult30F)
print 'viewRGbBayesResult: '
util.getIc(viewRGbBayesResult30F)
print 'CombinadosVotoMajoritarioResult: '
util.getIc(VMJTResult30F)

#Friedman Test
print "Friedman Test Para os 5 Classificadores para 99%% de Significancia"

Fvalue,pvalue,rankings,pivots = util.friedman_test(viewShapeKnnResult30F,viewRGBKnnResult30F,viewShapeBayesResult30F,viewRGbBayesResult30F,VMJTResult30F)
pivDic = {'knnV1':pivots[0], 'knnV2':pivots[1],'BayesV1': pivots[2],'BayesV2': pivots[3],'Vtmj': pivots[4]}

print 'p-value: ' 
print pvalue
print 'Rank: '
print pivDic

#Nemenyi Test

pivDic = {'knnV1':pivots[0], 'knnV2':pivots[1],'BayesV1': pivots[2],'BayesV2': pivots[3],'Vtmj': pivots[4]}

Comparions,Zvalues,pvalues,Adjustedpvalues = util.nemenyi_multitest(pivDic)

print "Friedman Test Para os 5 Ranks do teste de Friedman para os Classificadores com 99%% de Significancia:"

print "Comparacoes:"
print Comparions

print "P-Values Ajustados:"
print Adjustedpvalues
