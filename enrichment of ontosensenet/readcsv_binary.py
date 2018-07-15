import pickle
import re
import gensim
import pandas
from collections import Counter
import numpy
model = gensim.models.Word2Vec.load('corpus_v2_.word2vec')
wvec=model.wv
csv=pandas.read_csv("Final_verbs.csv")

wordcol = "క్రియ"
meancol = "అర్థం"
labels=[]
labels.append("To Know : 3")
labels.append("To Move : 4")
labels.append("To Do : 5 ")
labels.append("To Have : 6")
labels.append("To Be : 7")
labels.append("To Cut : 8 ")
labels.append("To Bound : 9")
#total 7 labelss

def compare(a,b):
	return len([i for i, j in zip(a, b) if i == j])*100.0/len(a)

def findlabel(word):
	if word in biglist:
		index=biglist.index(word)# gives first index, some words repaet, but lite
	else:
		for ind,i in enumerate(biglist):
			if i.replace("(","").replace(")","") == word:
				index=ind
			elif re.sub("[\(\[].*?[\)\]]", "", i) == word:
				index=ind
	for labelnumber,label in enumerate(labels):
		labelist=list(csv[label])
		if labelist[index]==1 or labelist[index]=="1":
			return labelnumber

#delete empty rows
csv.dropna(subset=[wordcol], inplace=True)

biglist=list(csv[wordcol]) #len 8574. 72 empty entteies. so. 8502
biglist=[re.sub("\d","",i) for i in biglist]#remove unnnessary numbers


# #how much present in word2vec
# w1=[1 if i in wvec else 0 for i in biglist]
# w1.count(1)
# #just 1342 in word2vec

# #remove brackers 
# W2=[a.replace("(","").replace(")","") for a in biglist]
# w2=[1 if i in wvec else 0 for i in W2]
# w2.count(1)
# #just 1352

# #remove bracets and remove xtra charachter
# W3=[re.sub("[\(\[].*?[\)\]]", "", a) for a in biglist]
# w3=[1 if i in wvec else 0 for i in W3]
# w3.count(1)
# wx=[]
# [wx.append(i) if w3]
# #jus 1430

#joining all
#final list of words
finalwords=[]
for i in biglist:
	if i in wvec:
		finalwords.append(i)
	else:
		if i.replace("(","").replace(")","") in wvec:
			finalwords.append(i.replace("(","").replace(")",""))
		if re.sub("[\(\[].*?[\)\]]", "", i) in wvec:
			finalwords.append(re.sub("[\(\[].*?[\)\]]", "", i))

finalwords=list(set(finalwords))
#just 1201
X=[model[word] for word in finalwords]
y=[findlabel(word) for word in finalwords]

###########################CLASSSSSSIFERSSSSSS##############
print("now starting classification")
#from tf_idf_features import dockeys
#dockeys=pickle.load(open("dockeys","r"))
#print len(dockeys)
#from tag_extractor import newlabeldict as labeldict
#from tag_extractor import newlabels as alllabels

#import pickle
import random
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

alllabels=[0,1,2,3,4,5,6]
vectors=X
labels=y
#docke

def compare(a,b):
	return len([i for i, j in zip(a, b) if i == j])*100.0/len(a)



#join labels and vectors
joinlv=list(zip(vectors,labels))

#shuffle
random.shuffle(joinlv)

#take first 250 as test
testdata=joinlv[0:250]
traindata=joinlv[250:len(joinlv)]


#check whether alltestlabels are in alltrainlabels
testlabels=set()
for i in testdata:
	testlabels.add(i[1])

trainlabels=set()
for i in traindata:
	trainlabels.add(i[1])

assert testlabels==trainlabels==set(alllabels)
#if assertion fails then do again shuffle


#if assertion passed, now get train, test, trainlabels, trainlabels
test=list(zip(*testdata))[0]
testlabels=list(zip(*testdata))[1]

train=list(zip(*traindata))[0]
trainlabels=list(zip(*traindata))[1]




#classification for each label
svmac={}
nbac={}
rfac={}
nnac={}
knnac={}
model={}
output={}
maxac={}
for label in alllabels:
	print(label)
	#create a binaryclassifier from trainlabel or testlabel
	test_binary_labels=[1 if label == testlabels[i] else 0 for i in range(len(testlabels))]
	#1 means label is there, 0 means it isnt there
	train_binary_labels=[1 if label == trainlabels[i] else 0 for i in range(len(trainlabels))]
	#FOR SVM 
	clfsvm = svm.SVC()
	clfsvm.fit(train,train_binary_labels)
	output1=list(clfsvm.predict(test))
	#compare output and test_bianry_labels
	svmac[label]=compare(output1,test_binary_labels)
	if max(output1.count(0),output1.count(1)) != len(output1):
		print(label,"_svm")
	#FOR KNN
	knn = KNeighborsClassifier(3)
	knn.fit(train,train_binary_labels)
	output2=list(knn.predict(test))
	#compare output and test_bianry_labels
	knnac[label]=compare(output2,test_binary_labels)
	if max(output2.count(0),output2.count(1)) != len(output2):
		print(label,"_knn")
	#FOR RANDOM FOREST
	rf = RandomForestClassifier(max_depth = 10)
	rf.fit(train,train_binary_labels)
	output3=list(rf.predict(test))
	#compare output and test_bianry_labels
	rfac[label]=compare(output3,test_binary_labels)
	if max(output3.count(0),output3.count(1)) != len(output3):
		print(label,"_rf")
	#FOR NEURAL NETWORKS
	#nn = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=( 100, 25, 7), random_state=1)
	nn.fit(train,train_binary_labels)
	output4=list(nn.predict(test))
	#compare output and test_binary_labels
	nnac[label]=compare(output4,test_binary_labels)
	if max(output4.count(0),output4.count(1)) != len(output4):
		print(label,"_nn")
	outputmax=[max(set(train_binary_labels), key=train_binary_labels.count)]*len(test)
	maxac[label]=compare(outputmax,test_binary_labels)
	output[label]=[output1,output2,output3,output4]
	model[label]=[clfsvm,knn,rf,nn]
	#ALL SAME ACCURACY (BENCHMARK)


#pickle.dump([svmac,rfac,knnac,nnac,maxac,output],open("results.pickle","w"))