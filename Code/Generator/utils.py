from __future__ import unicode_literals, print_function, division
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from load_data import *
from config import model_config as config
import spacy

nlp = spacy.load("en_core_web_lg")
from spacy.tokenizer import Tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab)

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2
PAD_token = 0
UNK_token = 3
class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
		self.n_words = 4  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)
	
	def addVocab(self, vocab):
		for word in vocab:
			self.addWord(word)


	def addWord(self, word):
		# should check the count and say if it is less than 3 then should be converted to UNK
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	# Turn a Unicode string to plain ASCII, thanks to
	# https://stackoverflow.com/a/518232/2809427
	def unicodeToAscii(s):
		return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		)

	# Lowercase, trim, and remove non-letter characters


	def normalizeString(s):
		s = Lang.unicodeToAscii(s.lower())
		s = Lang.unicodeToAscii(s.strip())
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
		return s


	def readLangs():
		print("Reading lines...")
		train_x = []
		valid_x = []
		test_x = []
		train_y = []
		train_y_reverse = []
		valid_y = []
		valid_y_reverse = []
		test_y = []
		test_y_reverse = []
		# Read train file
		train, valid, test = read_dataset(config['dataset'], config['level'])

		STYLE_ORDER = [config['style']]
		combined_dict_train, controlled_dict_train = aggregate_data(train, STYLE_ORDER, verbose=False)
		combined_dict_valid, controlled_dict_valid = aggregate_data(valid, STYLE_ORDER, verbose=False)
		combined_dict_test, controlled_dict_test = aggregate_data(test, STYLE_ORDER, verbose=False)

		if config['dataset_source'] == 'PASTEL':
			for style_order, value_dic in list(controlled_dict_train.items()):
			    for i in range(len(value_dic)):
			        if style_order[0] == 'Male' or style_order[0] == 'NoDegree' or style_order[0] == '55-74' or style_order[0] == '45-54':
			            train_y.append(0)
			            train_y_reverse.append(1)
			            train_x.append(Lang.normalizeString(value_dic[i]))
			        elif style_order[0] == 'Female' or style_order[0] == 'Bachelor' or style_order[0] == '25-34':
			            train_y.append(1)
			            train_y_reverse.append(0)
			            train_x.append(Lang.normalizeString(value_dic[i]))
			        '''else:
			            train_y.append(2)
			            train_x.append(Lang.normalizeString(value_dic[i]))'''
			for style_order, value_dic in list(controlled_dict_valid.items()):
			    for i in range(len(value_dic)):
			        if style_order[0] == 'Male' or style_order[0] == 'NoDegree' or style_order[0] == '55-74' or style_order[0] == '45-54':
			            valid_y.append(0)
			            valid_y_reverse.append(1)
			            valid_x.append(Lang.normalizeString(value_dic[i]))
			        elif style_order[0] == 'Female' or style_order[0] == 'Bachelor' or style_order[0] == '25-34':
			            valid_y.append(1)
			            valid_y_reverse.append(0)
			            valid_x.append(Lang.normalizeString(value_dic[i]))
			        '''else:
	                            valid_y.append(2)
	                            valid_x.append(Lang.normalizeString(value_dic[i]))'''
			for style_order, value_dic in list(controlled_dict_test.items()):
			    for i in range(len(value_dic)):
			        if style_order[0] == 'Male' or style_order[0] == 'NoDegree' or style_order[0] == '55-74' or style_order[0] == '45-54':
			            test_y.append(0)
			            test_y_reverse.append(1)
			            test_x.append(Lang.normalizeString(value_dic[i]))
			        elif style_order[0] == 'Female' or style_order[0] == 'Bachelor' or style_order[0] == '25-34':
			            test_y.append(1)
			            test_y_reverse.append(0)
			            test_x.append(Lang.normalizeString(value_dic[i]))
			        '''else:
	                            test_y.append(2)
	                            test_x.append(Lang.normalizeString(value_dic[i]))'''

		elif config['dataset_source'] == 'Yelp':
			train_x = open('../../Data/Yelp/sentiment.train.0', encoding='utf-8').read().split('\n')
			for i in range(len(train_x)):
				train_y.append(0)
				train_y_reverse.append(1)
			val = open('../../Data/Yelp/sentiment.train.1', encoding='utf-8').read().split('\n')
			for i in range(len(val)):
				train_x.append(val[i])
				train_y.append(1)
				train_y_reverse.append(0)
			test_x = open('../../Data/Yelp/sentiment.test.0', encoding='utf-8').read().split('\n')
			for i in range(len(train_x)):
				test_y.append(0)
				test_y_reverse.append(1)
			val = open('../../Data/Yelp/sentiment.test.1', encoding='utf-8').read().split('\n')
			for i in range(len(val)):
				test_x.append(val[i])
				test_y.append(1)
				test_y_reverse.append(0)
			valid_x = open('../../Data/Yelp/sentiment.dev.0', encoding='utf-8').read().split('\n')
			for i in range(len(valid_x)):
				valid_y.append(0)
				valid_y_reverse.append(1)
			val = open('../../Data/Yelp/sentiment.dev.1', encoding='utf-8').read().split('\n')
			for i in range(len(val)):
				valid_x.append(val[i])
				valid_y.append(1)
				valid_y_reverse.append(0)
		input_lang = Lang(config['style'])

		return input_lang, train_x, train_y, train_y_reverse, valid_x, valid_y, valid_y_reverse, test_x, test_y, test_y_reverse

def load_word_embeddings(embedding_type):

    if embedding_type == 'glove':
        embeddings_index = dict()
        f = open('../../../../Embeddings/Glove/'+config['ver'] + str(config['embedding_dim']) +'d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index
    else:
        return None

# given a text, returns a 2d matrix with its word embeddings.
# each column represents the embedding for each word
def get_embedding_matrix(embeddings_index, vocab, embedding_dim):
    oov = []
    if embeddings_index is not None:
        embedding_matrix = np.zeros((len(vocab)+4, embedding_dim))
        embedding_matrix[0] = np.zeros((1, embedding_dim), dtype='float32') #PAD
        embedding_matrix[1] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #SOS
        embedding_matrix[2] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #EOS
        embedding_matrix[3] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #UNK
        l = 4
        for key in vocab:
            embedding_vector = embeddings_index.get(key)
            if embedding_vector is not None:
                embedding_matrix[l] = embedding_vector
            else:
                oov.append(key)
                #embedding_matrix[l] = np.zeros((1, embedding_dim))
                #initializing it with zeros seems to work better
                embedding_matrix[l] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim))
            l += 1
        '''print("starting saving in oov file")
        with open('oov' + file_extension + '_glove.txt', 'w') as f:
            for item in oov:
                item = item.encode('utf8')
                f.write("%s\n" % item)
        f.close()
        print('len embedding matrix should be same as vocab')
        print(len(embedding_matrix))'''
        print(len(oov))
        return embedding_matrix
    else:
        embedding_matrix = np.zeros((len(vocab)+4, embedding_dim))
        embedding_matrix[0] = np.zeros((1, embedding_dim), dtype='float32') #PAD
        embedding_matrix[1] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #SOS
        embedding_matrix[2] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #EOS
        embedding_matrix[3] = np.random.uniform(low=-0.05, high=0.05, size=(1,embedding_dim)) #UNK
        l = 4
        for key in vocab:
        	embedding_matrix[l] = np.random.uniform(low=-1.0, high=1, size=(1,embedding_dim))
        	l += 1
        return embedding_matrix

def getvocab(sentences):
	inputvocab = []
	inputword2count = {}
	min_freq_inp = config['freq']
	for sent in sentences:
		for word in sent.split(' '):
			if word not in inputword2count:
				inputword2count[word] = 1
			else:
				inputword2count[word] += 1

	for k,v in inputword2count.items():
		if v >= min_freq_inp:
			inputvocab.append(k)
	return inputvocab


def filterPair(p):
	return len(p[0].split(' ')) < config['MAX_LENGTH']
		#p[1].startswith(eng_prefixes)


def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

def prepareData():
	input_lang, train_x, train_y, train_y_reverse, valid_x, valid_y, valid_y_reverse, test_x, test_y, test_y_reverse = Lang.readLangs()
	print("Read %s train sentence pairs" % len(train_x))
	print("Read %s valid sentence pairs" % len(valid_x))
	print("Read %s test sentence pairs" % len(test_x))
	#train_x = filterPairs(train_x)
	#valid_x = filterPairs(valid_x)
	#test_x = filterPairs(test_x)
	print("Counting words...")
	input_vocab = getvocab(train_x)
	input_lang.addVocab(input_vocab)
	embeddings_index = load_word_embeddings('glove')
	input_embedding_weights = get_embedding_matrix(embeddings_index, input_vocab, config['embedding_dim'])
	print("Counted words:")
	print(input_lang.name, input_lang.n_words)



	tag_vocab = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 
	'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 
	'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP', 
	'``']
	dep_vocab = ['ROOT', 'ACL', 'ACOMP', 'ADVCL', 'ADVMOD', 'AGENT', 'AMOD', 'APPOS', 'ATTR', 'AUX', 'AUXPASS', 
	'CASE', 'CC', 'CCOMP', 'COMPOUND', 'CONJ', 'CSUBJ', 'CSUBJPASS', 'DATIVE', 'DEP', 'DET', 'DOBJ', 'EXPL', 
	'INTJ', 'MARK', 'META', 'NEG', 'NMOD', 'NPADVMOD', 'NSUBJ', 'NSUBJPASS', 'NUMMOD', 'OPRD', 'PARATAXIS', 
	'PCOMP', 'POBJ', 'POSS', 'PRECONJ', 'PREDET', 'PREP', 'PRT', 'PUNCT', 'QUANTMOD', 'RELCL', 'XCOMP', '', 'SUBTOK']
	dep_lang = Lang('dep')
	tag_lang = Lang('tag')
	tag_lang.addVocab(tag_vocab)
	dep_lang.addVocab(dep_vocab)
	#tag_embedding_weights = get_embedding_matrix(None, tag_vocab, config['tag_dim'])
	#print(tag_embedding_weights.shape)
	#dep_embedding_weights = get_embedding_matrix(None, dep_vocab, config['dep_dim'])
	'''train_y = [[getword(input_lang, 'female')] if i else [getword(input_lang, 'male')] for i in train_y]
	valid_y = [[getword(input_lang, 'female')] if i else [getword(input_lang, 'male')] for i in valid_y]
	test_y = [[getword(input_lang, 'female')] if i else [getword(input_lang, 'male')] for i in test_y]'''
	#print(test_y)
	#return input_lang, output_lang, train_pairs, valid_pairs, test_pairs, [], []
	return input_lang, tag_lang, dep_lang, train_x, train_y, train_y_reverse, valid_x, valid_y, valid_y_reverse, test_x, test_y, test_y_reverse, input_embedding_weights

def add_noise(input_tensor, input_tensor_len):
	#print(input_tensor)
	#noisy = torch.zeros((input_tensor.shape[0], input_tensor.shape[1]), device = device, dtype=torch.int64)
	noisy = input_tensor.clone().detach()
	#return input_tensor
	for i in range(noisy.shape[0]):
		l = input_tensor_len[i].item()-1
		#a = noisy[i][1:l].clone().detach()
		#b=a.copy()
		#print(noisy[i])
		x = noisy[i][0:l].clone().detach()
		#print(x)
		#div = random.randint(4,6)
		div = 4
		b = x.cpu().numpy()
		for j in range(int(len(b)/div)):
			counter = 0
			while True:
				counter+= 1
				a = b[div*j:div*(j+1)].copy()
				c = a.copy()
				#a = b.copy()
				if counter >= 20:
					print(a)
					print(b)
					print('ha')
					break
				count = 0
				random.shuffle(a)
				#print(a)
				#print(b)
				for k in range(len(a)):
					#print(b)
					#print(a[i])
					#print(b.index(a[i]))
					if abs(list(c).index(a[k]) - k) <= 3:
						count += 1
					else:
						continue
				if count == len(a):
					b[div*j:div*(j+1)] = a
					break
		#print(a)
		#print(torch.from_numpy(a))
		#print('ye kya ha')
		#print(noisy[i])
		#print(noisy)
		noisy[i][0:l] = torch.from_numpy(b)
		#noisy[i][1:l] = torch.Tensor(torch.from_numpy(a), device=device, dtype=torch.int64)
		#print(noisy[i])
		#print(1/0)'''
		for k in range(noisy.shape[1]):
			if random.uniform(0,1) < 0.2 and noisy[i][k] != EOS_token and noisy[i][k] != SOS_token and noisy[i][k] != PAD_token:
				noisy[i][k] = 0
	#print(noisy)
	#print(1/0)
	return noisy


def convert_to_sent(sent):
	s = ''
	#print(sent)
	for i in range(len(sent)):
		s = s + sent[i] + ' '
	#print(s)
	return s[:-1]

def load_syntax_file(sentences, split_type):

	if os.path.isfile('Pos_'+split_type+'.txt') and os.path.isfile('Dep_'+split_type+'.txt'):
		print('syntax file present, loading')
		pos = open('Pos_'+split_type+'.txt', encoding='utf-8').read().split('\n')
		pos_sent = pos[:-1]
		dep = open('Dep_'+split_type+'.txt', encoding='utf-8').read().split('\n')
		print(len(dep))
		dep_sent = dep[:-1]
		print(len(dep_sent))

	else:
		pos_sent = []
		dep_sent = []
		print('syntax file not present, creating')
		with open('Pos_'+split_type+'.txt', "a") as pos:
			with open('Dep_'+split_type+'.txt', "a") as dep:

				for i in range(len(sentences)):
					doc=nlp(sentences[i])
					a = convert_to_sent([(tok.dep_).upper() for tok in doc])
					dep_sent.append(a)
					dep.write(a + "\n")
					a = convert_to_sent([(tok.tag_).upper() for tok in doc])
					pos_sent.append(a)
					pos.write(a + "\n")

	return pos_sent, dep_sent

def pad_sequences(x, max_len):
    padded = torch.zeros((max_len), dtype=torch.long, device=device)
    if len(x) > max_len: padded[:] = x[:max_len]
    else:
    	padded[:len(x)] = x
    return padded

def getword(lang, word):
	if word in lang.word2index:
		return lang.word2index[word]
	else:
		return UNK_token #index number of UNK

def indexesFromSentence(sentence, lang):
	return [getword(lang, word) for word in sentence.split(' ')]


def tensorFromSentence(sentence, lang):
    indexes = indexesFromSentence(sentence, lang)
    indexes.append(EOS_token)
    #indexes = [SOS_token] + indexes
    return torch.tensor(indexes, dtype=torch.long, device=device)

def tensorsFromPair(sentence, lang):
    input_tensor = tensorFromSentence(sentence, lang)
    return input_tensor

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def get_len(itensor):
    a = []
    mask = []
    for i in itensor:
        count = 0
        temp = []
        #print(i)
        for j in i:
            if j == 0:
                break
            else:
                count += 1
        a.append(count)
    return torch.tensor(a, device=device)

def get_mask(tensor_len):
	a = torch.max(tensor_len)
	t = []
	for i in tensor_len:
		temp = []
		for k in range(i):
			temp.append(1)
		for k in range(a-i):
			temp.append(0)
		t.append(temp)
	return torch.tensor(t, device=device)
