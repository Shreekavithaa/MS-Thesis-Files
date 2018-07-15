import pickle
import re
import os
import gensim
import pandas
from collections import Counter
import numpy
from nltk import word_tokenize
#"__________________________" is delimiter
#Book reviews
def compare(a,b):
	return len([i for i, j in zip(a, b) if i == j])*100.0/len(a)


bp=open("Book Reviews/book_pos.txt").read()
bn=open("Book Reviews/book_neg.txt").read()

bp=bp.split("__________________________")
bn=bn.split("__________________________")
if '' in bp:
	bp.remove('')
if '' in bn:
	bn.remove('')
bp=[i.strip().replace("\n"," ") for i in bp]
bn=[i.strip().replace("\n"," ") for i in bn]
reviews=bp+bn
labels=[1]*len(bp)+[0]*len(bn) 	#1 postitice 0 negative
print(len(reviews))
#Movie reviews

bp=open("Movie Reviews/pos_review").read()
bn=open("Movie Reviews/neg_review").read()

bp=bp.split("__________________________")
bn=bn.split("__________________________")
if '' in bp:
	bp.remove('')
if '' in bn:
	bn.remove('')
bp=[i.strip().replace("\n"," ") for i in bp]
bn=[i.strip().replace("\n"," ") for i in bn]
reviews=reviews+bp+bn
labels=labels+[1]*len(bp)+[0]*len(bn) 	#1 postitice 0 negative
print(len(reviews))
#Product reviews

bp=open("Product Reviews/product_pos.txt").read()
bn=open("Product Reviews/product_neg.txt").read()

bp=bp.split("__________________________")
bn=bn.split("__________________________")
if '' in bp:
	bp.remove('')
if '' in bn:
	bn.remove('')
bp=[i.strip().replace("\n"," ") for i in bp]
bn=[i.strip().replace("\n"," ") for i in bn]
reviews=reviews+bp+bn
labels=labels+[1]*len(bp)+[0]*len(bn) 	#1 postitice 0 negative
print(len(reviews))

#Song reviews
for i in os.listdir("./Song_Lyrics/positive"):
	bp=open("./Song_Lyrics/positive/"+i).read()
	bp=bp.replace("\n"," ")
	reviews.append(bp)
	labels.append(1)


for i in os.listdir("./Song_Lyrics/negative"):
	bn=open("./Song_Lyrics/negative/"+i).read()
	bn=bn.replace("\n"," ")
	reviews.append(bn)
	labels.append(0)

print(len(reviews))

##################################
from sklearn.model_selection import train_test_split
import random 

joinlv=list(zip(reviews,labels))
#shuffle
random.shuffle(joinlv)
reviews=list(zip(*joinlv))[0]
labels=list(zip(*joinlv))[1]
rev_train, rev_test, lab_train, lab_test = train_test_split(reviews, labels, test_size=.3, random_state=42)
rev_test=reviews
lab_test=labels
#####load sentiwordnet (unigram data)

up=open("sentiwordnet/TE_POS.txt").read()
un=open("sentiwordnet/TE_NEG.txt").read()

t=up.split("##############################################################################")
up=t[5]
t=un.split("##############################################################################")
un=t[5]



up=re.split(r"[A-Za-z]",up)
up=[i.strip().replace("_","") for i in up]


un=re.split(r"[A-Za-z]",un)
un=[i.strip().replace("_","") for i in un]


##########Classfocato
from nltk import word_tokenize
from copy import deepcopy

ltest=deepcopy(lab_test)
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	lis1=[1 if j in up else 0 for j in x]
	lis2=[-1 if j in un else 0 for j in x]
	print(len(x),lis1.count(1),lis2.count(-1))
	if lis1.count(1)>lis2.count(-1):
		y.append(1)
	elif lis1.count(1)==lis2.count(-1):
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))


# ltest=deepcopy(lab_test)
# y=[]
# for k,i in enumerate(rev_test):
# 	x=word_tokenize(i)
# 	lis1=[1 if j in bp else 0 for j in x]
# 	lis2=[-1 if j in bn else 0 for j in x]
# 	print(len(x),lis1.count(1),lis2.count(-1))
# 	if lis1.count(1)<lis2.count(-1):1
# 		y.append(0)
# 	else:
# 		y.append(1)
# print(compare(y,lab_test))

###load birgram data
import pandas
from ast import literal_eval

csv=pandas.read_csv("bigrams.csv")
#csv.dropna(subset=["bigrams"], inplace=True)
bp=[]
bn=[]
ba=[]
for j,i in enumerate(csv["exist"]):
	if i==1:#sentiment exists
		if csv["p"][j]==1:
			bp.append(csv["bigrams"][j])
		if csv["n"][j]==1:
			bn.append(csv["bigrams"][j])
		if csv["a"][j]==1:
			ba.append(csv["bigrams"][j])

bp=[literal_eval(i) for i in bp]
bn=[literal_eval(i) for i in bn]
ba=[literal_eval(i) for i in ba]


#####  bi gram classificaton
from nltk import word_tokenize
from copy import deepcopy

ltest=deepcopy(lab_test)
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	x=list(zip(x,x[1:]))
	lis1=[1 if j in bp else 0 for j in x]
	lis2=[-1 if j in bn else 0 for j in x]
	print(len(x),lis1.count(1),lis2.count(-1))
	if lis1.count(1)>lis2.count(-1):
		y.append(1)
	elif lis1.count(1)==lis2.count(-1):
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))


####### bigram + unigram

from nltk import word_tokenize
from copy import deepcopy

ltest=deepcopy(lab_test)
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	lis1=[1 if j in up else 0 for j in x]
	lis2=[-1 if j in un else 0 for j in x]
	ux=list(zip(x,x[1:]))
	bis1=[1 if j in bp else 0 for j in ux]
	bis2=[-1 if j in bn else 0 for j in ux]
	pcount=lis1.count(1)+bis1.count(1)
	ncount=lis2.count(-1)+bis2.count(-1)
	print(len(x),pcount,ncount)
	if pcount>ncount:
		y.append(1)
	elif pcount==ncount:
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))
