import pickle
import re
import os
import gensim
import pandas
from collections import Counter
import numpy
import tqdm
from nltk import word_tokenize
from copy import deepcopy
#"__________________________" is delimiter
#Book reviews
def compare(a,b):
	return len([i for i, j in zip(a, b) if i == j])*100.0/len(a)

#తనుకామిస్తే
import sys                                                                                                                             
sys.path.insert(0, '/home/anvesh/Downloads/Softwares/anoopkunchukuttan-indic_nlp_library-INDIC_NLP_0.3-59-g7614520/anoopkunchukuttan-indic_nlp_library-7614520/src')

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.morph import unsupervised_morph 
from indicnlp import common

common.INDIC_RESOURCES_PATH="/home/anvesh/Downloads/Softwares/anoopkunchukuttan-indic_nlp_library-INDIC_NLP_0.3-59-g7614520/anoopkunchukuttan-indic_nlp_library-7614520/src/indicnlp/resources"
analyzer=unsupervised_morph.UnsupervisedMorphAnalyzer('te')

# input_text="प्रधानमंत्री नरेंद्र मोदी ने इंडोनेशिया की आजादी के संघर्ष के शहीदों को श्रद्धांजलि देकर अपनी यात्रा की शुरुआत की. उन्होंने यहां कलीबाता नेशनल हीरोज सिमेट्री में शहीदों को पुष्पांजलि अर्पित की.".decode('utf-8')
# 		   #प्रधानमंत्री नरेंद्र मोदी ने इंडोनेशिया की आजादी के संघर्ष के शहीदों को श्रद्धांजलि देकर अपनी यात्रा की शुरुआत की. उन्होंने यहां कलीबाता नेशनल हीरोज सिमेट्री में शहीदों को पुष्पांजलि अर्पित की.


# analyzer=unsupervised_morph.UnsupervisedMorphAnalyzer('mr')

# #			  రవీందర్ సింగ్ నవల ఈ టూ హడ్ ఎ లవ్ స్టోరి ఇది ప్రేమ కథ కానీ బోరింగ్ ఈ పుస్తకం కొనకండి నేను ఈ పుస్తకాన్ని చదివాను చాలా పెద్ద కధ కాని ఆసక్తికరంగా లేదు  సగం కథ పూర్తయింది లేదా బాలుడు మరియు బాలిక  కుషి  ఒకరికొకరు కలవరు  300 పేజీలు పుస్తకం నన్ను నమ్మండి దయచేసి కొని మీ డబ్బు వృథా చేసుకోకండి 
# 			  #రవీందర్ సింగ్ నవల ఈ టూ హడ్ ఎ లవ్ స్టో రి ఇది ప్రేమ కథ కానీ బో రింగ్ ఈ పుస్తకం కొన కండి నే ను ఈ పుస్తక ాన్ని చదివా ను చాలా పెద్ద కధ కాని ఆసక్తికర ంగా లేదు  సగం కథ పూర్తయి ంది లేదా బాల ుడు మరియు బాలిక  కు షి  ఒకరి కొక రు కలవర ు  300 పేజీ లు పుస్తకం నన్న ు నమ్మ ండి దయ చేసి కొని మీ డబ్బు వృథా చేసుకో కండి 

# # రవీందర్ సింగ్ నవల

# indic_string=u'రవీందర్ సింగ్ నవల'

#STEM SENTIRAMA DATA

[reviews,labels]=pickle.load(open("reviews_labels.pickle","rb"))
[up,un,bp,bn,uap,uan]=pickle.load(open("sentilists.pickle","rb"))
reviews=list(reviews)
reviews.pop(336)
labels=list(labels)
labels.pop(336)

stemreviews=[]
#STEM REVIWES
for j,review in tqdm.tqdm(enumerate(reviews)):
	analyzes_tokens=analyzer.morph_analyze_document(word_tokenize(review))
	stemreviews.append(" ".join(analyzes_tokens))
#STEM LISTS

#stem unigram

sup=[analyzer.morph_analyze_document(word_tokenize(i))[0] for i in up]
sun=[analyzer.morph_analyze_document(word_tokenize(i))[0] for i in un]
suap=[analyzer.morph_analyze_document(word_tokenize(i))[0] for i in uap]
suan=[analyzer.morph_analyze_document(word_tokenize(i))[0] for i in uan]
sbp=[analyzer.morph_analyze_document(i) for i in bp]
sbn=[analyzer.morph_analyze_document(i) for i in bn]

def startswith(ch,st):
	return st[0]==ch

def find(charlist,c1):
	for j,i in enumerate(charlist):
		if i==c1:
			return j

def find2(charlist,c2,ind):
	for j,i in enumerate(charlist):
		if i==c2 and j!=ind:
			return j


def pick(bgrm,stemlist):
	c1=bgrm[0][0]
	c2=bgrm[1][0]
	ct=0
	ct1=0
	firstchars=[i[0] for i in stemlist]
	j=find(firstchars,c1)
	k=find2(firstchars,c2,j)
	s=deepcopy(stemlist)
	for ii,stem in enumerate(stemlist):
		if ii!=j and ii!=k:
			s[ii]=0 
	return s

ssbp=[pick(bp[j],stemlist) for j,stemlist in enumerate(sbp)]
ssbn=[pick(bn[j],stemlist) for j,stemlist in enumerate(sbn)]
for i in ssbp:
	while i[-1]==0:
		i.pop(-1)

for i in ssbn:
	while i[-1]==0:
		i.pop(-1)


def fit(review,sbg):
	ct=0
	# print(sbg)
	review=review
	x=word_tokenize(review)
	for j,term in enumerate(x):
		if term==sbg[0]:
			if sbg[-1] in x[j:j+sbg.count(0)+2]:
				ct=ct+1
	return ct

#true stemming bigram
import random
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

rev_train, rev_test, lab_train, lab_test = train_test_split(reviews, labels, test_size=.3, random_state=42)
rev_test=reviews
lab_test=labels

revbpcnt=[]
revbncnt=[]
for i in tqdm.tqdm(reviews):
	revbpcnt.append(sum([fit(i,k) for k in ssbp]))
	revbncnt.append(sum([fit(i,k) for k in ssbn]))


from nltk import word_tokenize
from copy import deepcopy

ltest=deepcopy(lab_test)
y=[]
for k,i in enumerate(rev_test):
	if revbpcnt[k]>revbncnt[k]:
		y.append(1)
	elif revbpcnt[k]==revbncnt[k]:
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))



# indic_string=u'రవీందర్ సింగ్ నవల ఈ టూ హడ్ ఎ లవ్ స్టోరి ఇది ప్రేమ కథ కానీ బోరింగ్ ఈ పుస్తకం కొనకండి నేను ఈ పుస్తకాన్ని చదివాను చాలా పెద్ద కధ కాని ఆసక్తికరంగా లేదు  సగం కథ పూర్తయింది లేదా బాలుడు మరియు బాలిక  కుషి  ఒకరికొకరు కలవరు  300 పేజీలు పుస్తకం నన్ను నమ్మండి దయచేసి కొని మీ డబ్బు వృథా చేసుకోకండి '
# analyzes_tokens=analyzer.morph_analyze_document(indic_string.split(' '))

# for i in analyzes_tokens:
# 	print(i.decode("utf-8"))


# remove_nuktas=True
# factory=IndicNormalizerFactory()
# normalizer=factory.get_normalizer("hi",remove_nuktas)
# print normalizer.normalize(input_text)


# आपल्या
# हिरड्या
# च्या
# आणि
# दाता
# च्या
# मध्ये
# जीवाणू
# असतात
# .


##########Classfocato
from nltk import word_tokenize
from copy import deepcopy

ltest=deepcopy(lab_test)
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	lis1=[1 if j in sup else 0 for j in x]
	lis2=[-1 if j in sun else 0 for j in x]
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

ltest=lab_test
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	lis1=[1 if j in sup else 0 for j in x]
	lis2=[-1 if j in sun else 0 for j in x]
	lis1=lis1+[1 if j in suap else 0 for j in x]
	lis2=lis2+[-1 if j in suan else 0 for j in x]
	print(len(x),lis1.count(1),lis2.count(-1))
	if lis1.count(1)>lis2.count(-1):
		y.append(1)
	elif lis1.count(1)==lis2.count(-1):
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))

###ALL
ltest=lab_test
y=[]
for k,i in enumerate(rev_test):
	x=word_tokenize(i)
	lis1=[1 if j in sup else 0 for j in x]
	lis2=[-1 if j in sun else 0 for j in x]
	lis1=lis1+[1 if j in suap else 0 for j in x]
	lis2=lis2+[-1 if j in suan else 0 for j in x]
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
	lis1=[1 if j in sup else 0 for j in x]
	lis2=[-1 if j in sun else 0 for j in x]
	lis1=lis1+[1 if j in suap else 0 for j in x]
	lis2=lis2+[-1 if j in suan else 0 for j in x]
	pcount=lis1.count(1)+revbpcnt[k]
	ncount=lis2.count(-1)+revbncnt[k]
	print(len(x),pcount,ncount)
	if pcount>ncount:
		y.append(1)
	elif pcount==ncount:
		y.append(-100)
	else:
		y.append(0)

print(len([i for i, j in zip(y, ltest) if i == j])*100.0/(len(ltest)-y.count(-100)))
print(compare(y,ltest))
 