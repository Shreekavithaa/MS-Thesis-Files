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

import random
from sklearn import svm
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

alllabels=[0,1,2,3,4,5,6]
vectors=X
labels=y

#join labels and vectors
joinlv=list(zip(vectors,labels))


#shuffle
random.shuffle(joinlv)

#take first 800 as test
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
svmaccuracy={}
nbaccuracy={}
rfaccuracy={}
nnaccuracy={}
model={}
output={}


clfsvm = svm.SVC(verbose=True)
clfsvm.fit(train,trainlabels)
output1=list(clfsvm.predict(test))

#compare output and test_bianry_labels
svmaccuracy=compare(output1,testlabels)
print(precision_recall_fscore_support(testlabels, output1, average='macro'))
print(precision_recall_fscore_support(testlabels, output1, average='micro'))
print(precision_recall_fscore_support(testlabels, output1, average='weighted'))

# #FOR NAIVE BAYES
# clfbayes = MultinomialNB()
# clfbayes.fit(train,trainlabels)
# output2=list(clfbayes.predict(test))

# #compare output and test_bianry_labels
# nbaccuracy=compare(output2,testlabels)
# if max(output2.count(0),output2.count(1)) != len(output2):
# 	print label + "_nb"



#FOR RANDOM FOREST
rf = RandomForestClassifier(max_depth = 10)
rf.fit(train,trainlabels)
output3=list(rf.predict(test))

#compare output and test_bianry_labels
rfaccuracy=compare(output3,testlabels)
if max(output3.count(0),output3.count(1)) != len(output3):
	print(label + "_rf")
	

#FOR NEURAL NETWORKS
#nn = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(train,trainlabels)
output4=list(nn.predict(test))

#compare output and test_bianry_labels
nnaccuracy[label]=compare(output4,testlabels)
if max(output4.count(0),output4.count(1)) != len(output4):
	print label + "_nn"




#output[label]=[output1,output2,output3]
#model[label]=[clfsvm,clfbayes,rf]

#CLUSTERING
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)

import sklearn_extensions as ske
mdl = ske.fuzzy_kmeans.FuzzyKMeans()
mdl.fit(X)
