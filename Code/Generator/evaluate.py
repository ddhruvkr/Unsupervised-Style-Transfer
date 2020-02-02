from utils import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from model.SARI import calculate
from config import model_config as config
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score,precision_score, classification_report, precision_recall_fscore_support


sf = SmoothingFunction()
def evaluate(encoder, decoder, sentence, label, attribute, input_lang, output_lang, p, max_length=config['MAX_LENGTH']):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence, input_lang)
        input_tensor = pad_sequences(input_tensor, max_length)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(1, True)
        input_tensor = input_tensor.unsqueeze(0)
        #print(input_tensor)
        c = 0
        a = []
        for i in input_tensor[0]:
            if i.item() == 0:
                a.append(0)
            else:
                c += 1
                a.append(1)
        input_tensor_len = torch.tensor([c], device=device)
        #print(input_tensor_len)
        #input_tensor_len = get_len(input_tensor)
        input_tensor_mask = get_mask(input_tensor_len)
        #print(input_tensor_mask)
        encoder_output, encoder_hidden = encoder(input_tensor, input_tensor_len, encoder_hidden)
        batch_size = 1
        #decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
        #print(decoder_input)
        #print(decoder_input.shape)
        decoder_input = attribute.unsqueeze(0).to(device)
        #decoder_input = torch.tensor(attribute.unsqueeze(0), device=device, dtype=torch.int64)
        #print(decoder_input)
        #print(decoder_input.shape)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            _, decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, input_tensor_mask)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.shape)
            #print(topv)
            #print(topi)
            #print(input_tensor_mask)
            #print(input_tensor_mask.item())
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            elif topi.item() != EOS_token:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze(0).detach()
            #print(decoder_input)
        return decoded_words, decoder_attentions[:di + 1]

def evaluateBest(pairs, p):
    candidate = []
    reference = []
    n = min(1300, len(pairs))
    for i in range(n):
        pair = pairs[i]
        if p:
            print('>', pair[0])
            print('=', pair[1])
        candidate.append(pair[0].split(' '))
        reference.append([pair[1].split(' ')])
    print('Cumulative 4-gram: %f' % corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))

def evaluateSARI(encoder, decoder, pairs, input_lang, output_lang, p):
    candidate = []
    reference = []
    source = []
    total_score = 0
    n = min(1300, len(pairs))
    for i in range(n):
        pair = pairs[i]
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, p)
        output_sentence = ' '.join(output_words[:-1])
        #output_sentence = pair[0]
        if p:
            print('>', pair[0])
            print('=', pair[1])
            print('<', output_sentence)
        source.append(pair[0])
        with open("output.txt", "a") as file:
            file.write(output_sentence + "\n")
        candidate.append(output_sentence)
        reference.append([pair[1]])
        #print(source[0])
        #print(candidate[0])
        #print(reference[0])
        score,_,_,_ = calculate(source[0], candidate[0], reference[0])
        total_score += score
        #print(score, end=" ")
    print(total_score/n)

def accuracy(out, labels):
    #outputs = np.argmax(out, axis=1)
    val,ind = out.max(1)
    a = torch.sum(ind == labels)
    f1 = f1_score(labels.cpu(), ind.cpu(), average='macro')
    return f1, a.float()/ind.shape[0]


def evaluate_classifier(model, x, xp, xd, y, criterion, from_test):
    
    for param in model.parameters():
        param.requires_grad = False
    predictions = model(x,xp,xd).squeeze(1)
    predictions = predictions.double()
    if from_test:
        #predictions = torch.sigmoid(predictions)
        print(predictions)
        return 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    loss = criterion(predictions, y)
    f1, acc = accuracy(predictions, y)
    #r,p,f,r1,p1,f1 = calculate_precision_recall_f1(p_concat.cpu(), y_concat.cpu(), model_name)
    #return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_score/len(iterator)
    return loss, acc, f1


def evaluateBLUE(encoder, decoder, pairs, labels, attributes, input_lang, output_lang, p, bleu_vector):
    candidate = []
    reference = []
    n = min(5000, len(pairs))
    for i in range(n):
        p = False
        if i < 10:
            p = True
        pair = pairs[i]
        if p:
            print(labels[i])
            print('>', pair)
        output_words, attentions = evaluate(encoder, decoder, pair, labels[i], attributes[i], input_lang, output_lang, p)
        '''if p:
            print([pair.split(' ')])
            print(output_words[:-1])'''
        candidate.append(output_words[:-1])
        reference.append([pair.split(' ')])
        output_sentence = ' '.join(output_words)
        if p:
            print('<', output_sentence)
            print('')
    '''if p:
        print(reference)
        print(candidate)'''
    sc = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
    print('Cumulative 4-gram: %f' % sc)
    bleu_vector.append(sc)


