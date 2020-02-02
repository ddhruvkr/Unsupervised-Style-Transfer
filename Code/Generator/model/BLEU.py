# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
sf = SmoothingFunction()
reference = [['this', 'is', 'small', 'test'],['this', 'is', 'test']]
candidate = ['this', 'is', 'a', 'test']
reference1 = [[['this', 'is', 'small', 'test'],['this', 'is', 'test']], [['i', 'ate', 'an', 'apple']]]
candidate1 = [['this', 'is', 'a', 'test'], ['ate', 'an', 'potato']]
reference1 = [[['this', 'is', 'small', 'test'],['this', 'is', 'test']], [['i', 'ate', 'an', 'apple']]]
candidate1 = [['this', 'is', 'a', 'test'], ['ate', 'an', 'potato']]
#score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))
print('Cumulative 4-gram: %f' % corpus_bleu(reference1, candidate1, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))
#print(score)

#https://www.nltk.org/_modules/nltk/translate/bleu_score.html