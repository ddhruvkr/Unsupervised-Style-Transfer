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

all_data = open('data-simplification/newsela_article_corpus_2016-01-29/newsela_data.txt', encoding='utf-8').read().split('\n')
adata = open('data-simplification/newsela_article_corpus_2016-01-29/final.txt', encoding='utf-8').read().split('\n')
zhang = open('zhang.txt', encoding='utf-8').read().split('\n')
my = {}
his = {}
count = 0
for d in zhang:
	his[d] = 1
l = len(adata)
for ind, data in enumerate(adata):
	if ind < l-1:
		d = data.split('\t')
		if d[3] not in his:
			print(d[3])
			count += 1
print(count)
		#print(d[3])
'''with open('data-simplification/newsela_article_corpus_2016-01-29/cleaned.txt', 'w') as f:
	for index, data in enumerate(all_data):
		d = data.split('\t')
		src_ver = d[1]
		dst_ver = d[2]
		if ((src_ver == 'V0' and dst_ver == 'V1') or (src_ver == 'V1' and dst_ver == 'V2') or (src_ver == 'V2' and dst_ver == 'V3')):
			continue
		else:
			f.write("%s\n" % data)
f.close()

count = 0
for data in adata:
	d = data.split('\t')'''