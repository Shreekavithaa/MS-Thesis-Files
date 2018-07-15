import gensim, logging,os
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname, ngrams=3):
        self.dirname = dirname
        #self.tokenizer = RegexpTokenizer(r'\w+')
        self.ngrams = ngrams
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            ct=0
            for line in open(os.path.join(self.dirname, fname)):
                line.replace("=","")
                #print(line)
                #t=ct+1
                #print(ct)
                tokens = word_tokenize(line)
                #tokens = self.tokenizer.tokenize(line)
                yield [" ".join(tokens[i:i+self.ngrams]) for i in range(len(tokens)-self.ngrams)]

sentences = MySentences('./skp/',1)
model = gensim.models.Word2Vec(sentences,window=5,min_count=1,size=300,workers=4)
model.save('corpus_v2_.word2vec')

#import gensim
#model = gensim.models.Word2Vec.load('corpus_v2_.word2vec')
