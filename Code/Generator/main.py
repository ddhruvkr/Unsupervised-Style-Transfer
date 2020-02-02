from utils import *
from model.encoder import EncoderRNN
from model.decoder import DecoderRNN
from model.decoder_attn import AttnDecoderRNN
from config import model_config as config
#from model.decoder_attn_pointer import PAttnDecoderRNN
from test import *
from model.Attn_one_hot import WordAttention
from evaluate import *

#input_lang, output_lang, train_pairs, valid_pairs, test_pairs, input_embedding_weights, output_embedding_weights = prepareData(embedding_dim, freq, ver, dataset)

input_lang, tag_lang, dep_lang, x_train, y_train, y_train_reverse, x_valid, y_valid, y_valid_reverse, x_test, y_test, y_test_reverse, embedding_weights = prepareData()
#print(random.choice(train_pairs))
attribute_train = []
attribute_valid = []
attribute_test = []
label_test = []
label_train = []
label_valid = []
a1 = 'positive'
#female
a2 = 'negative'
#male
style_tok_train = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_train]
style_tok_train_reverse = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_train_reverse]
style_tok_valid = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_valid]
style_tok_valid_reverse = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_valid_reverse]
style_tok_test = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_test]
style_tok_test_reverse = [[getword(input_lang, a1)] if i else [getword(input_lang, a2)] for i in y_test_reverse]
attribute_train.append(torch.LongTensor(style_tok_train))
attribute_train.append(torch.LongTensor(style_tok_train_reverse))
attribute_valid.append(torch.LongTensor(style_tok_valid))
attribute_valid.append(torch.LongTensor(style_tok_valid_reverse))
attribute_test.append(torch.LongTensor(style_tok_test))
attribute_test.append(torch.LongTensor(style_tok_test_reverse))
label_train.append(torch.LongTensor(y_train))
label_train.append(torch.LongTensor(y_train_reverse))
label_valid.append(torch.LongTensor(y_valid))
label_valid.append(torch.LongTensor(y_valid_reverse))
label_test.append(torch.LongTensor(y_test))
label_test.append(torch.LongTensor(y_test_reverse))
print(input_lang.n_words)
#print(len(train_pairs))
encoder = EncoderRNN(input_lang.n_words, config['hidden_size'], config['num_layers'], embedding_weights, config['embedding_dim'], config['dropout']).to(device)
decoder1 = DecoderRNN(config['hidden_size'], input_lang.n_words, config['MAX_LENGTH'], config['num_layers'], embedding_weights, config['embedding_dim'], config['dropout']).to(device)
#decoder1 = AttnDecoderRNN(config['hidden_size'], input_lang.n_words, config['MAX_LENGTH'], config['num_layers'], embedding_weights, config['embedding_dim'], config['dropout']).to(device)
decoder2 = AttnDecoderRNN(config['hidden_size'], input_lang.n_words, config['MAX_LENGTH'], config['num_layers'], embedding_weights, config['embedding_dim'], config['dropout']).to(device)
classifier = model = WordAttention(input_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['hidden_size'], config['classifer_class_size'], config['num_layers'], config['dropout'], embedding_weights, config['structural'])
classifier.load_state_dict(torch.load(config['classifier_name'] + '.pt'))
def init_weights(m):
    for name, param in m.named_parameters():
        if name != 'embedding.weight':
            if 'weight' in name:
                #print(name)
                #print(param)
                #nn.init.uniform_(param.data, -0.1, 0.1)
                nn.init.normal_(param.data, mean=0, std=0.1)
            else:
                nn.init.constant_(param.data, 0)
#print('main me 0.1')            
#encoder.apply(init_weights)
#decoder.apply(init_weights)
trainIters(encoder, decoder1, classifier, x_train, label_train, attribute_train, x_valid, label_valid, attribute_valid, x_test, label_test, attribute_test, input_lang, input_lang)

#evaluateBLUE(encoder, decoder, test_pairs, input_lang, output_lang, False)

#evaluateBest(test_pairs, False)

#evaluateSARI(encoder, decoder, test_pairs, input_lang, output_lang, False)
