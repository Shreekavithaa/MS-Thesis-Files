import pickle
import re
import os
import gensim
import pandas
import tqdm
from collections import Counter
import numpy
from nltk import word_tokenize
#"__________________________" is delimiter
#Book reviews
def compare(a,b):
	return len([i for i, j in zip(a, b) if i == j])*100.0/len(a)

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def mostsimilar(word):
	a=model.most_similar(word.decode('utf-8'))
	r=[]
	for i in a:
		print(i[0])
		newword=i[0].encode('utf-8')
		r.append((newword,i[1]))
	return r		

def vec(word):
	return model[word.decode('utf-8')]

def similarity(a,b):
	return model.similarity(a.decode('utf-8'),b.decode('utf-8'))

import gensim
#model = gensim.models.KeyedVectors.load_word2vec_format("./corpus.v3.bin",unicode_errors='ignore',binary=True)
#model

[reviews,labels]=pickle.load(open("reviews_labels.pickle","rb"))
[up,un,bp,bn,uap,uan]=pickle.load(open("sentilists.pickle","rb"))


a="రాపాడించడంలో"

# X=model.vocab
# vectors=[]
# for j,i in ennumerate(reviews):
# 	x=word_tokenize(i)
# 	v=numpy.array([0]*200)
# 	for word in x:
# 		if word in X:
# 			v=v+model[word]
# 	v=v/(1.0*len(x))
# 	vectors.append(v)

vectors=list(pickle.load(open("vectors.pickle","rb")))



def isNaN(num):
	return num != num

for j,i in enumerate(vectors):
	if isNaN(i[0])==True:
		print(j)

#336 is the problem 
vectors.pop(336)
reviews=list(reviews)
reviews.pop(336)
labels=list(labels)
labels.pop(336)
# ##Classfication
# joinlv=list(zip(vectors,labels))
# #shuffle
# random.shuffle(joinlv)
# vectors=list(zip(*joinlv))[0]
# labels=list(zip(*joinlv))[1]
# [train,test,trainlabels,testlabels]=train_test_split(vectors,labels,test_size=.5,random_state=42)
# testlabels=testlabels
# #1 means label is there, 0 means it isnt there
# trainlabels=trainlabels
# #FOR SVM 
# clfsvm = svm.SVC()
# clfsvm.fit(train,trainlabels)
# output1=list(clfsvm.predict(test))
# #compare output and test_bianry_labels
# svmac1=compare(output1,testlabels)


#now modified vectors also
vectors1=[]
for k,review in tqdm.tqdm(enumerate(reviews)):
	x=word_tokenize(review)
	posuni=[1 if j in up else 0 for j in x].count(1)
	neguni=[-1 if j in un else 0 for j in x].count(-1)
	posbi=[1 if j in bp else 0 for j in x].count(1)
	negbi=[-1 if j in bn else 0 for j in x].count(-1)
	posann=[1 if j in uap else 0 for j in x].count(1)
	negann=[-1 if j in uan else 0 for j in x].count(-1)
	newvector=numpy.concatenate((vectors[k],[posuni,neguni,posbi,negbi,posann,negann]),axis=0)
	#newvector=numpy.concatenate((vectors[k],[posuni,neguni,posbi,negbi,posann,negann]),axis=0)
	vectors1.append(newvector)







import random
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt



def Classify(vectors, vectors1, labels):
	#join labels and vectors
	joinlv=list(zip(vectors,vectors1,labels))
	#shuffle
	random.shuffle(joinlv)
	vectors=list(zip(*joinlv))[0]
	vectors1=list(zip(*joinlv))[1]
	labels=list(zip(*joinlv))[2]
	[train,test,trainlabels,testlabels]=train_test_split(vectors,labels,test_size=.3,random_state=42)
	#take first 250 as test
	#if assertion passed, now get train, test, trainlabels, trainlabels
	#classification for eac1h label
	svmac1={}
	nbac1={}
	rfac1={}
	nnac1={}
	knnac1={}
	model={}
	output={}
	maxac1={}
	#Linear SVM
	clfsvm = svm.LinearSVC()
	clfsvm.fit(train,trainlabels)
	output0=list(clfsvm.predict(test))
	#compare output and test_bianry_labels
	lsvmac1=compare(output0,testlabels)
	if max(output0.count(0),output0.count(1)) != len(output0):
		print("_linear_svm")
	#FOR SVM 
	clfsvm = svm.SVC()
	clfsvm.fit(train,trainlabels)
	output1=list(clfsvm.predict(test))
	#compare output and test_bianry_labels
	svmac1=compare(output1,testlabels)
	if max(output1.count(0),output1.count(1)) != len(output1):
		print("_svm")
	#FOR KNN
	knn = KNeighborsClassifier(3)
	knn.fit(train,trainlabels)
	output2=list(knn.predict(test))
	#compare output and test_bianry_labels
	knnac1=compare(output2,testlabels)
	if max(output2.count(0),output2.count(1)) != len(output2):
		print("_knn")
	#FOR RANDOM FOREST
	rf = RandomForestClassifier(max_depth = 10)
	rf.fit(train,trainlabels)
	output3=list(rf.predict(test))
	#compare output and test_bianry_labels
	rfac1=compare(output3,testlabels)
	if max(output3.count(0),output3.count(1)) != len(output3):
		print("_rf")
	#FOR NEURAL NETWORKS
	#nn = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=( 100, 25, 7), random_state=1)
	nn.fit(train,trainlabels)
	output4=list(nn.predict(test))
	#compare output and testlabels
	nnac1=compare(output4,testlabels)
	if max(output4.count(0),output4.count(1)) != len(output4):
		print("_nn")
	outputmax=[max(set(trainlabels), key=trainlabels.count)]*len(test)
	maxac1=compare(outputmax,testlabels)
	output=[output1,output2,output3,output4]
	model=[clfsvm,knn,rf,nn]
	ac1=[lsvmac1, svmac1,rfac1,nnac1,knnac1,maxac1]
	[train,test,trainlabels,testlabels]=train_test_split(vectors1,labels,test_size=.3,random_state=42)
	print("lsvm svm rf nn knn max")
	print(ac1)
		#take first 250 as test
	#if assertion passed, now get train, test, trainlabels, trainlabels
	#classification for eac1h label
	svmac2={}
	nbac2={}
	rfac2={}
	nnac2={}
	knnac2={}
	model={}
	output={}
	maxac2={}
	#Linear SVM
	clfsvm = svm.LinearSVC()
	clfsvm.fit(train,trainlabels)
	output0=list(clfsvm.predict(test))
	#compare output and test_bianry_labels
	lsvmac2=compare(output0,testlabels)
	if max(output0.count(0),output0.count(1)) != len(output0):
		print("_linear_svm")
	#FOR SVM 
	clfsvm = svm.SVC()
	clfsvm.fit(train,trainlabels)
	output1=list(clfsvm.predict(test))
	#compare output and test_bianry_labels
	svmac2=compare(output1,testlabels)
	if max(output1.count(0),output1.count(1)) != len(output1):
		print("_svm")
	#FOR KNN
	knn = KNeighborsClassifier(3)
	knn.fit(train,trainlabels)
	output2=list(knn.predict(test))
	#compare output and test_bianry_labels
	knnac2=compare(output2,testlabels)
	if max(output2.count(0),output2.count(1)) != len(output2):
		print("_knn")
	#FOR RANDOM FOREST
	rf = RandomForestClassifier(max_depth = 10)
	rf.fit(train,trainlabels)
	output3=list(rf.predict(test))
	#compare output and test_bianry_labels
	rfac2=compare(output3,testlabels)
	if max(output3.count(0),output3.count(1)) != len(output3):
		print("_rf")
	#FOR NEURAL NETWORKS
	#nn = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=( 100, 25, 7), random_state=1)
	nn.fit(train,trainlabels)
	output4=list(nn.predict(test))
	#compare output and testlabels
	nnac2=compare(output4,testlabels)
	if max(output4.count(0),output4.count(1)) != len(output4):
		print("_nn")
	outputmax=[max(set(trainlabels), key=trainlabels.count)]*len(test)
	maxac2=compare(outputmax,testlabels)
	output=[output1,output2,output3,output4]
	model=[clfsvm,knn,rf,nn]
	ac2=[lsvmac2, svmac2,rfac2,nnac2,knnac2,maxac2]
	print("linsvc svm rf nn knn max")
	print(ac2)
	return[ac1,ac2]
	#ALL SAME ac1CURac1Y (BENCHMARK)




#plain Vanila classification



[[svmac1,nbac1,rfac1,nnac1,knnac1,maxac1],[svmac2,nbac2,rfac2,nnac2,knnac2,maxac2]] = Classify(vectors,vectors1,labels)	
print("classification1 done")




#l[0].encode('utf-8')==l[0]
#s=l[0].encode('utf-8')

#a="తనుకామిస్తే"
# తనుకామిస్తే


#సౌజన్య
def mostsimilar(word):
	a=model.most_similar(word.decode('utf-8'))
	r=[]
	for i in a:
		print(i[0])
		newword=i[0].encode('utf-8')
		r.append((newword,i[1]))
	return r		

def vec(word):
	return model[word.decode('utf-8')]


mostsimilar('సౌజన్య')

###clustering

from sklearn.cluster import KMeans
import numpy as np
vectors1=vectors
joinlv=list(zip(vectors,vectors1,labels))
random.shuffle(joinlv)
v=list(zip(*joinlv))[0]
v1=list(zip(*joinlv))[1]
labels=list(zip(*joinlv))[2]
[train,test,trainlabels,testlabels]=train_test_split(v,labels,test_size=.3,random_state=42)

X=np.array(train)	
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
	kmeans.labels_
# kmeans.predict([[0, 0], [4, 4]])
kmeans.cluster_centers_
