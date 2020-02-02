from config import model_config as config
from utils import *
from test_structured import *

input_lang, tag_lang, dep_lang, x_train, x_valid, x_test, embedding_weights, tag_embedding_weights, dep_embedding_weights = prepareData()

'''x_pos_train, x_dep_train = load_syntax_file(x_train, "train_binary_gender")
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
x_dep_test = [pad_sequences(x, config['MAX_LENGTH']) for x in dep_tensor_test]'''

if config['lm_type'] == 'structural':
    from model.structural_decoder import DecoderGRU
    from test_structured import *
    from evaluate_structured import *
elif config['lm_type'] == 'standard':
    from model.decoder import DecoderRNN
    from test import *
    from evaluate import *


#print(len(train_pairs))
if config['lm_type'] == 'structural':
    decoder = DecoderGRU(config['hidden_size'], input_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
        embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)
    train_pos, train_dep = load_syntax_file(x_train, 'train_binary_gender')
    valid_pos, valid_dep = load_syntax_file(x_valid, 'valid_binary_gender')
    test_pos, test_dep = load_syntax_file(x_test, 'test_binary_gender')
    print('loaded pos, dep files')
elif config['lm_type'] == 'standard':
    decoder = DecoderRNN(config['hidden_size'], input_lang.n_words, config['num_layers'], 
        embedding_weights, config['embedding_dim'], config['dropout']).to(device)

if config['lm_type'] == 'structural':
    trainIters(decoder, x_train, x_valid, x_test, input_lang, tag_lang, dep_lang, train_pos, train_dep, valid_pos, valid_dep, test_pos, test_dep)
elif config['lm_type'] == 'standard':
    trainIters(decoder, train_simple_unique, valid_simple_unique, test_simple_unique, input_lang)
#decoder.load_state_dict(torch.load('lm_forward_all.pt'))
#evaluatePerplexity(decoder, test_simple_unique, output_lang, False)
