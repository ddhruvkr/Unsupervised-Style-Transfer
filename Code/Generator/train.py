import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *
from config import model_config as config
from evaluate import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: decrease this ratio as training proceeds

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train(batch_size, input_tensor, input_labels, input_attributes, input_tensor_mask, input_tensor_len, 
    target_tensor, target_labels, target_attributes, target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
    encoder_optimizer, decoder_optimizer, classifer_optimizer, criterion, criterion_classifier, 
    input_lang, include_loss_from_classifier):
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden(batch_size, False)
    
    input_length = input_tensor.size(1)
    #target_length = config['MAX_LENGTH']
    target_length = torch.max(target_tensor_len).item()
    loss = 0
    loss_classifer = 0

    #print(input_tensor.shape)
    #32,75 batch_size, max_length
    encoder_output, encoder_hidden = encoder(input_tensor, input_tensor_len, encoder_hidden)
    #decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
    #print(input_attribute)
    #decoder_input = torch.tensor(input_attributes, device=device, dtype=torch.int64)
    decoder_input = target_attributes.to(device)
    #print(decoder_input)
    output = torch.zeros((config['MAX_LENGTH'],config['batch_size'], input_lang.n_words), device=device, dtype=torch.float64)
    decoder_hidden = encoder_hidden
    for di in range(target_length):
        #print(di)
        gs, decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, input_tensor_mask)
        indices = torch.tensor([di], device=device)
        #print(target_tensor)
        #print(indices)
        target_t = torch.index_select(target_tensor, 1, indices)
        #print(target_t)
        #print(1/0)
        #topv, topi = decoder_output.topk(1)
        #decoder_input = topi.squeeze(0).detach()
        loss += criterion(decoder_output[0], target_t.view(-1))
        #print(decoder_output.shape)
        #print(gs)
        #print(gs.shape)
        decoder_input = target_t
        output[di] = decoder_output[0]
    #max_len, bts, vocab
    #print(output.shape)
    output = output.permute(1,0,2)
    classifier_input = output.exp().float()
    #print(classifier_input)
    #topv, topi = output.topk(1)
    #topi = topi.squeeze(2)
    #print(topi)
    #print(topi.shape)
    f1_score_classifier = 0
    acc_classifier = 0
    #for i in range(topi.shape[0]):
    #    for k in range(top)
    if include_loss_from_classifier:
        #input_labels  = target_labels.to(device)
        target_labels = target_labels.to(device)
        #topi = topi.to(device)
        classifier = classifier.to(device)
        criterion_classifier = criterion_classifier.to(device)
        loss_classifer, acc_classifier, f1_score_classifier = evaluate_classifier(classifier, classifier_input, None, None, target_labels, criterion_classifier, False)
        #print(acc_classifier)
    #topv, topi = output.topk(1)
    #topi = topi.squeeze(2)
    #print(topi.shape)
    return output, loss, loss_classifer, acc_classifier




def evaluation_calculate(batch_size, input_tensor, input_labels, input_attributes, input_tensor_mask, input_tensor_len, 
    target_tensor, target_labels, target_attributes, target_tensor_mask, target_tensor_len, encoder, decoder, classifier, 
    criterion, criterion_classifier, input_lang, include_loss_from_classifier):

    encoder.eval()
    decoder.eval()
    encoder_hidden = encoder.initHidden(batch_size, False)
    
    input_length = input_tensor.size(1)
    #target_length = config['MAX_LENGTH']
    target_length = torch.max(target_tensor_len).item()
    loss = 0
    loss_classifer = 0

    #print(input_tensor.shape)
    #32,75 batch_size, max_length
    encoder_output, encoder_hidden = encoder(input_tensor, input_tensor_len, encoder_hidden)
    #decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
    #print(input_attribute)
    #decoder_input = torch.tensor(input_attributes, device=device, dtype=torch.int64)
    decoder_input = target_attributes.to(device)
    #print(decoder_input)
    output = torch.zeros((config['MAX_LENGTH'],config['batch_size'], input_lang.n_words), device=device, dtype=torch.float64)
    decoder_hidden = encoder_hidden
    for di in range(target_length):
        #print(di)
        gs, decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, input_tensor_mask)
        indices = torch.tensor([di], device=device)
        #print(target_tensor)
        #print(indices)
        target_t = torch.index_select(target_tensor, 1, indices)
        #print(target_t)
        #print(1/0)
        #topv, topi = decoder_output.topk(1)
        #decoder_input = topi.squeeze(0).detach()
        output[di] = decoder_output[0]
        loss += criterion(decoder_output[0], target_t.view(-1))
        decoder_input = target_t  # Teacher forcing
        #if decoder_input.item() == 0:
        #    break 
    #max_len, bts, vocab
    #print(output.shape)
    output = output.permute(1,0,2)
    classifier_input = output.exp().float()
    #print(topi)
    #print(topi.shape)
    f1_score_classifier = 0
    acc_classifier = 0
    #for i in range(topi.shape[0]):
    #    for k in range(top)
    if include_loss_from_classifier:
        #input_labels  = target_labels.to(device)
        target_labels = target_labels.to(device)
        #topi = topi.to(device)
        classifier = classifier.to(device)
        criterion_classifier = criterion_classifier.to(device)
        loss_classifer, acc_classifier, f1_score_classifier = evaluate_classifier(classifier, classifier_input, None, None, target_labels, criterion_classifier, False)
        #print(acc_classifier)
    #print(topi.shape)
    # bts, max_len (so that this could be the new input)
    #encoder_output, encoder_hidden = encoder(topi, input_tensor_len, encoder_hidden)
    '''for di in range(target_length):
        indices = torch.tensor([di], device=device)
        target_t = torch.index_select(target_tensor, 1, indices)
        loss += criterion(output[di],target_t.view(-1))'''
    '''else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, input_tensor_mask)
            indices = torch.tensor([di], device=device)
            target_t = torch.index_select(target_tensor, 1, indices)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(0).detach()  # detach from history as input
            loss += criterion(decoder_output[0], target_t.view(-1))
            #if decoder_input.item() == EOS_token:
            #    break'''
    return output, loss, loss_classifer, acc_classifier
    
