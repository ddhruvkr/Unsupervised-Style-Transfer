import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score,precision_score, classification_report, precision_recall_fscore_support
from torch.nn import functional as f
from model.Attn import WordAttention
from config import model_config as config

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
	def __init__(self, x_train, x_pos_train, x_dep_train, y_train):
		self.x_train = x_train
		self.x_pos_train = x_pos_train
		self.x_dep_train = x_dep_train
		self.y_train = y_train.to(device)

	def __len__(self):
		return len(self.x_train)

	def __getitem__(self, index):
		x = self.x_train[index]
		xp = self.x_pos_train[index]
		xd = self.x_dep_train[index]
		y = self.y_train[index]

		return x, xp, xd, y

def load_data(dataset, batch_size):
	dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
	return dataloader

def accuracy(out, labels):
    #outputs = np.argmax(out, axis=1)
    val,ind = out.max(1)
    a = torch.sum(ind == labels)
    f1 = f1_score(labels.cpu(), ind.cpu(), average='macro')
    return f1, a.float()/ind.shape[0]

def train(model, iterator, optimizer, criterion, epoch_no, input_lang):
	epoch_loss = 0
	epoch_acc = 0
	f1_score = 0
	model.train()
	i = 0
	max_val = 200
	total = len(iterator)
	for x,xp,xd,y in iterator:
		i += 1
		optimizer.zero_grad()
		#print(x.shape)
		x = torch.nn.functional.one_hot(x, input_lang.n_words).float()
		#x_ = torch.unsqueeze(x, 2)
		#one_hot = Variable(torch.cuda.FloatTensor(x.shape[0], x.shape[1], input_lang.n_words).zero_())
		#one_hot.scatter_(2, x_, 1)
		#print(x.shape)
		predictions= model(x,xp,xd).squeeze(1)
		predictions = predictions.double()
		#print(predictions)
		#print(y)
		loss = criterion(predictions, y)
		f1, acc = accuracy(predictions, y)
		#p, r, f1 = calculate_precision_recall_f1(predictions, y)
		loss.backward()
		#clip_gradient(model, 5e-1)
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc
		f1_score += f1
		if i % max_val == 0:
			print(f'| Epoch: {epoch_no+1} | Iter: {i} of {total} Train Loss: {(epoch_loss/i):.3f} | Train Acc: {epoch_acc/i*100:.2f}% | F1 score: {f1_score/i*100:.2f}%')

	print('train done')
	#calculate_precision_recall_f1(p_concat, y_concat)
	return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_score/len(iterator)

def evaluate(model, iterator, criterion, model_name, from_test, input_lang):
    
	epoch_loss = 0
	epoch_acc = 0
	f1_score = 0
	model.eval()
	with torch.no_grad():
		i = 0
		for x,xp,xd,y in iterator:
			i += 1
			x = torch.nn.functional.one_hot(x, input_lang.n_words).float()
			predictions = model(x,xp,xd).squeeze(1)
			predictions = predictions.double()
			if from_test:
				#predictions = torch.sigmoid(predictions)
				print(predictions)
				return 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
			loss = criterion(predictions, y)
			#predictions = torch.sigmoid(predictions)
			if i > 1:
				p_concat = torch.cat((p_concat, predictions), 0)
				y_concat = torch.cat((y_concat, y), 0)
			else:
				p_concat = predictions
				y_concat = y

	f1, acc = accuracy(p_concat, y_concat)
	epoch_loss += loss.item()
	epoch_acc += acc
	f1_score += f1
	#r,p,f,r1,p1,f1 = calculate_precision_recall_f1(p_concat.cpu(), y_concat.cpu(), model_name)
	#return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_score/len(iterator)
	return epoch_loss, epoch_acc, f1_score

def clip_gradient(model, clip_value):
	params = list(filter(lambda p: p.grad is not None, model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value, clip_value)

def build_and_train_network(x_train, x_pos_train, x_dep_train, y_train, 
	x_validation, x_pos_validation, x_dep_validation, y_validation,
	x_test, x_pos_test, x_dep_test, y_test, 
	input_lang, tag_lang, dep_lang, embedding_weights):
	model_name = config['model']
	print(device)
	print(input_lang.n_words)
	if model_name == 'Attn':
		model = WordAttention(input_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['hidden_size'], config['classifer_class_size'], config['num_layers'], config['dropout'], embedding_weights, config['structural'])
	is_cuda = True
	training_set = Dataset(x_train, x_pos_train, x_dep_train, y_train)
	train_iterator = load_data(training_set, config['batch_size'])
	validation_set = Dataset(x_validation, x_pos_validation, x_dep_validation, y_validation)
	validation_iterator = load_data(validation_set, config['batch_size'])
	test_set = Dataset(x_test, x_pos_test, x_dep_test, y_test)
	test_iterator = load_data(test_set, config['batch_size'])
	optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-6)
	'''weights = [3.0, 1.0]
	weights = torch.DoubleTensor(weights)
	if is_cuda:
		weights = weights.cuda()'''
	criterion = nn.NLLLoss()
	#criterion = nn.BCEWithLogitsLoss()
	try:
		model = model.to(device)
	except:
		print('trying 2nd time')
		model = model.to(device)
	criterion = criterion.to(device)
	validf1 = 0
	for epoch in range(config['epochs']):
		train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion, epoch, input_lang)
		print("validation accuracy")
		valid_loss, valid_acc, valid_f1 = evaluate(model, validation_iterator, criterion, model_name, False, input_lang)
		test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion, model_name, False, input_lang)
		#new_avg_f1 = (train_f1 + test_f1)/2.0
		#since the validation set is very small compared to the test set just using the validation loss is unstable
		if validf1 < valid_f1:
			validf1 = valid_f1
			print('saving model')
			torch.save(model.state_dict(), config['classifier_name'])
		#print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
		print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | F1 Score: {train_f1*100:.2f}% |  Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | F1 Score: {valid_f1*100:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | F1 Score: {test_f1*100:.2f}%')
	model.load_state_dict(torch.load(config['classifier_name']))
	print("test accuracy")
	
	#return test_p, test_r, test_f1, test_macro_p, test_macro_r, test_macro_f1

def test(x_test, y_test, z, vocab, embedding_weights, 
	word_sequence_length, emb_dim, hidden_dim, lr,  model, epochs, lstm_sizes, batch_size, 
	dropout_prob):
	model_name = model
	if model == 'CoAttn':
		model = IntraAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('IA.pt'))
	if model == 'BiRNN':
		model = BiRNN(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('BiRNN.pt'))
	if model == 'Attn':
		model = WordAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('HAN.pt'))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	is_cuda = False
	if device.type == 'cuda':
		is_cuda = True
	test_set = Dataset(x_test.astype(np.int64), y_test, z, is_cuda)
	test_iterator = load_data(test_set, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-6)
	'''weights = [3.0, 1.0]
	weights = torch.DoubleTensor(weights).cuda()'''
	criterion = nn.BCEWithLogitsLoss()
	try:
		model = model.to(device)
	except:
		print('trying 2nd time')
		model = model.to(device)
	criterion = criterion.to(device)
	test_loss, test_acc, test_r, test_p, test_f1, test_macro_r, test_macro_p, test_macro_f1 = evaluate(model, test_iterator, criterion, model_name, True)
	print(test_acc)
