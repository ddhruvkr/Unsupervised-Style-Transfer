import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from utils import *
from train import *
from evaluate import *
from config import model_config as config

class Dataset(data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, a, b, c, d, e):
        #'Initialization'
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.a)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample

        # Load data and get label
        a = self.a[index]
        b = self.b[index]
        c = self.c[index]
        d = self.d[index]
        e = self.e[index]

        return a,b,c,d,e

def load_data(dataset, batch_size):
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader

def trainIters(encoder, decoder, classifier, x_train, label_train, attribute_train, x_valid, label_valid, attribute_valid, x_test, label_test, attribute_test, input_lang, output_lang, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    criterion = nn.NLLLoss(ignore_index=0)
    criterion_classifier = nn.NLLLoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config['lr'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config['lr'])
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=config['lr'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-6)
    # TODO: use better optimizer
    x_pair = []
    for i in range(len(x_train)):
        pair = tensorsFromPair(x_train[i], input_lang)
        x_pair.append(pair)
    x_pair = [pad_sequences(x, config['MAX_LENGTH']) for x in x_pair]
    #y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
    training_set = Dataset(x_pair, label_train[0], label_train[1], attribute_train[0], attribute_train[1])
    training_iterator = load_data(training_set, config['batch_size'])
    lambda_AE = 1.0
    bleu_reconstruction = []
    blue_style_transfer = []
    accuracy_reconstruction = []
    accuracy_style_transfer = []
    for epoch in range(config['epochs']):
        print('epoch')
        print(epoch+1)
        lambda_AE -= 0.000005
        i = 0
        f1_org = 0
        f1_switched = 0
        for input_tensor, labels, labels_reverse, attributes, attributes_reverse in training_iterator:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_tensor_len = get_len(input_tensor)
            input_tensor_mask = get_mask(input_tensor_len)
            input_tensor_noisy = add_noise(input_tensor, input_tensor_len)
            target_tensor = input_tensor
            target_tensor_len = input_tensor_len
            target_tensor_mask = input_tensor_mask
            # Denoising Autoencoder -> but no style transfer
            _, loss_1, loss_c, _ = train(config['batch_size'], input_tensor_noisy, labels, attributes, 
                input_tensor_mask, input_tensor_len, target_tensor, labels, attributes, 
                target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
                encoder_optimizer, decoder_optimizer, classifier_optimizer, criterion, criterion_classifier, 
                input_lang, False)

            #(lambda_AE*loss_1/config['MAX_LENGTH']).backward()
            #(lambda_AE*loss_1).backward()
            #loss_1/config['MAX_LENGTH'].backward()
            #clip_gradient(encoder, 100e-1)
            #clip_gradient(decoder, 100e-1)

            #encoder_optimizer.step()
            #decoder_optimizer.step()

            #encoder_optimizer.zero_grad()
            #decoder_optimizer.zero_grad()

            
            # Switch from given style to new style
            '''output, _, loss_c12, f1_s = train(config['batch_size'], input_tensor.detach(), labels.detach(), attributes.detach(), 
                input_tensor_mask.detach(), input_tensor_len.detach(), target_tensor.detach(), labels_reverse.detach(), attributes_reverse.detach(), 
                target_tensor_mask.detach(), target_tensor_len.detach(), encoder, decoder, classifier, 
                encoder_optimizer, decoder_optimizer, classifier_optimizer, criterion, criterion_classifier, 
                input_lang, True)'''
            output, _, loss_c12, f1_s = train(config['batch_size'], input_tensor, labels, attributes, 
                input_tensor_mask, input_tensor_len, target_tensor, labels_reverse, attributes_reverse, 
                target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
                encoder_optimizer, decoder_optimizer, classifier_optimizer, criterion, criterion_classifier, 
                input_lang, True)

            #print(loss_c12)
            #loss_c12.backward(retain_graph=True)

            #clip_gradient(encoder, 100e-1)
            #clip_gradient(decoder, 100e-1)

            #encoder_optimizer.step()
            #decoder_optimizer.step()

            #encoder_optimizer.zero_grad()
            #decoder_optimizer.zero_grad()

            topv, topi = output.topk(1)
            input_tensor_switch = topi.squeeze(2)
            input_tensor_len = get_len(input_tensor_switch)
            input_tensor_mask = get_mask(input_tensor_len)
            # Take generated sentence and switch back to original style
            input_tensor_backtranslate, loss_2, loss_c21, f1_o = train(config['batch_size'], input_tensor_switch, labels_reverse, attributes_reverse, 
                input_tensor_mask, input_tensor_len, target_tensor, labels, attributes, 
                target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
                encoder_optimizer, decoder_optimizer, classifier_optimizer, criterion, criterion_classifier, 
                input_lang, True)
            #ll = loss_2/config['MAX_LENGTH'] + loss_c12
            #print(loss_2)
            #print(loss_2.data)
            #loss_2.data = loss_2.data + loss_c12.data*config['MAX_LENGTH']
            #print(loss_2)
            #print(loss_2.data)
            #loss_2 = loss_2.item() + loss_c12.item()*config['MAX_LENGTH']
            #loss_c12.backward()
            if config['classifier_loss']:
                ll = lambda_AE*loss_1 + loss_2 + 0.2*loss_c12
            else:
                ll = lambda_AE*loss_1 + loss_2
            #ll = loss_2
            ll.backward()
            #loss_2.backward()
            #clip_gradient(encoder, 100e-1)
            #clip_gradient(decoder, 100e-1)

            encoder_optimizer.step()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            #loss_total = (lambda_AE/(epoch+1) * loss_1) + loss_2
            #print(loss_c12)
            #print(loss_1/config['MAX_LENGTH'])
            #print(loss_2/config['MAX_LENGTH'])
            #print('ha')
            loss_total = loss_1 + loss_2 + loss_c12
            f1_switched += f1_s
            f1_org += f1_o


            '''loss_total.backward()
            clip_gradient(encoder, 100e-1)
            clip_gradient(decoder, 100e-1)

            encoder_optimizer.step()
            decoder_optimizer.step()'''
            target_length = torch.max(target_tensor_len).item()
            loss = loss_total.item() / target_length
            i += 1
            print_loss_total += loss
            plot_loss_total += loss
            #print(loss)
            if i % 100 == 0:
                #print(print_loss_total)
                print_loss_avg = print_loss_total / 100
                print_loss_total = 0
                iters = epoch*len(x_train) + (config['batch_size']*i)
                n_iters = config['epochs']*len(x_train)
                print(loss_c12*config['MAX_LENGTH'])
                print(loss_1)
                print(loss_2)
                print('%s (%d%%) %.4f %.4f %.4f' % (timeSince(start, iters/n_iters),
                                             iters / n_iters * 100, print_loss_avg, f1_org / i, f1_switched / i ))


        #showPlot(plot_losses)

        #evaluate_metrics(encoder, decoder, classifier, x_train, label_train, attribute_train, input_lang, output_lang, False)
        #evaluate_metrics(encoder, decoder, classifier, x_valid, label_valid, attribute_valid, input_lang, output_lang, False)
        evaluate_metrics(encoder, decoder, classifier, x_test, label_test, attribute_test, input_lang, output_lang, False, True, accuracy_reconstruction)
        evaluate_metrics(encoder, decoder, classifier, x_test, label_test, attribute_test, input_lang, output_lang, False, False, accuracy_style_transfer)
        #evaluateBLUE(encoder, decoder, x_train, label_train[0], attribute_train[0], input_lang, output_lang, False)
        #evaluateBLUE(encoder, decoder, x_valid, label_valid[0], attribute_valid[0], input_lang, output_lang, False)
        evaluateBLUE(encoder, decoder, x_test, label_test[0], attribute_test[0], input_lang, output_lang, False, bleu_reconstruction)
        evaluateBLUE(encoder, decoder, x_test, label_test[1], attribute_test[1], input_lang, output_lang, False, blue_style_transfer)
        #evaluateSARI(encoder, decoder, pairs, input_lang, output_lang, False)
        #evaluateSARI(encoder, decoder, valid_pairs, input_lang, output_lang, False)
        #evaluateSARI(encoder, decoder, test_pairs, input_lang, output_lang, False)
        print(accuracy_reconstruction)
        print(accuracy_style_transfer)
        print(bleu_reconstruction)
        print(blue_style_transfer)






def evaluate_metrics(encoder, decoder, classifier, x_train, label_train, attribute_train, 
    input_lang, output_lang, include_loss_from_classifier, reverse, accuracy_vector, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    criterion = nn.NLLLoss(ignore_index=0)
    criterion_classifier = nn.NLLLoss()
    
    # TODO: use better optimizer
    x_pair = []
    for i in range(len(x_train)):
        pair = tensorsFromPair(x_train[i], input_lang)
        x_pair.append(pair)
    x_pair = [pad_sequences(x, config['MAX_LENGTH']) for x in x_pair]
    #y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
    training_set = Dataset(x_pair, label_train[0], label_train[1], attribute_train[0], attribute_train[1])
    training_iterator = load_data(training_set, config['batch_size'])
    avg_acc = 0.0
    itr = 0
    for input_tensor, labels, labels_reverse, attributes, attributes_reverse in training_iterator:
        itr += 1
        input_tensor_len = get_len(input_tensor)
        input_tensor_mask = get_mask(input_tensor_len)
        target_tensor = input_tensor
        target_tensor_len = input_tensor_len
        target_tensor_mask = input_tensor_mask
        # Denoising Autoencoder -> but no style transfer
        if reverse:
            _, _, _, acc = evaluation_calculate(512, input_tensor, labels, attributes, 
            input_tensor_mask, input_tensor_len, target_tensor, labels, attributes, 
            target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
            criterion, criterion_classifier, input_lang, True)
        else:
            _, _, _, acc = evaluation_calculate(512, input_tensor, labels, attributes, 
                input_tensor_mask, input_tensor_len, target_tensor, labels_reverse, attributes_reverse, 
                target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
                criterion, criterion_classifier, input_lang, True)

        avg_acc += acc
        #print(loss)
    print((avg_acc.item())/itr)
    accuracy_vector.append((avg_acc.item())/itr)