from config import model_config as config
from utils import *
from classification import *

input_lang, tag_lang, dep_lang, x_train, y_train, x_valid, y_valid, x_test, y_test, embedding_weights,tag_embedding_weights, dep_embedding_weights = prepareData()

x_pos_train, x_dep_train = load_syntax_file(x_train, "train_binary_gender")
x_pos_valid, x_dep_valid = load_syntax_file(x_valid, "valid_binary_gender")
x_pos_test, x_dep_test = load_syntax_file(x_test, "test_binary_gender")


tensor_train = []
pos_tensor_train = []
dep_tensor_train = []
tensor_valid = []
pos_tensor_valid = []
dep_tensor_valid = []
tensor_test = []
pos_tensor_test = []
dep_tensor_test = []

print(len(x_train))
print(len(y_train))
for i in range(len(x_train)):
    tensor_train.append(tensorFromSentence(x_train[i], input_lang))
    pos_tensor_train.append(tensorFromSentence(x_pos_train[i], tag_lang))
    dep_tensor_train.append(tensorFromSentence(x_dep_train[i], dep_lang))

for i in range(len(x_valid)):
    tensor_valid.append(tensorFromSentence(x_valid[i], input_lang))
    pos_tensor_valid.append(tensorFromSentence(x_pos_valid[i], tag_lang))
    dep_tensor_valid.append(tensorFromSentence(x_dep_valid[i], dep_lang))

for i in range(len(x_test)):
    tensor_test.append(tensorFromSentence(x_test[i], input_lang))
    pos_tensor_test.append(tensorFromSentence(x_pos_test[i], tag_lang))
    dep_tensor_test.append(tensorFromSentence(x_dep_test[i], dep_lang))

x_train = [pad_sequences(x, config['MAX_LENGTH']) for x in tensor_train]
x_pos_train = [pad_sequences(x, config['MAX_LENGTH']) for x in pos_tensor_train]
x_dep_train = [pad_sequences(x, config['MAX_LENGTH']) for x in dep_tensor_train]
x_valid = [pad_sequences(x, config['MAX_LENGTH']) for x in tensor_valid]
x_pos_valid = [pad_sequences(x, config['MAX_LENGTH']) for x in pos_tensor_valid]
x_dep_valid = [pad_sequences(x, config['MAX_LENGTH']) for x in dep_tensor_valid]
x_test = [pad_sequences(x, config['MAX_LENGTH']) for x in tensor_test]
x_pos_test = [pad_sequences(x, config['MAX_LENGTH']) for x in pos_tensor_test]
x_dep_test = [pad_sequences(x, config['MAX_LENGTH']) for x in dep_tensor_test]
y_train = torch.LongTensor(y_train)
y_valid = torch.LongTensor(y_valid)
y_test = torch.LongTensor(y_test)
#print(y_train.shape)

#print(y_train.shape)
#y_train = F.one_hot(torch.LongTensor(y_train), config['classifer_class_size']).to(torch.double)
#y_valid = F.one_hot(torch.LongTensor(y_valid), config['classifer_class_size']).to(torch.double)
#y_test = F.one_hot(torch.LongTensor(y_test), config['classifer_class_size']).to(torch.double)

for i in range(1):
	print('Iteration: ', i)
	build_and_train_network(x_train, x_pos_train, x_dep_train, y_train, x_valid, x_pos_valid, x_dep_valid, 
		y_valid, x_test, x_pos_test, x_dep_test, y_test, input_lang, tag_lang, dep_lang, embedding_weights)
