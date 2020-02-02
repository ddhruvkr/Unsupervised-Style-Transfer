import random
import math
import numpy
import torch

x = torch.rand((60))
#a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
x1 = x[1:40].clone().detach()
b=x1.numpy()
print(b)

print(len(b))
for k in range(int(len(b)/5)):
	while True:
		a = b[5*k:5*(k+1)].copy()
		c = a.copy()
		#print(a)
		#break
		count = 0
		random.shuffle(a)
		#print(a)
		#print(b)
		for i in range(len(a)):
			#print(b)
			#print(a[i])
			#print(b.index(a[i]))
			#numpy.where(b == a[i])
			if abs(list(c).index(a[i]) - i) <= 3:
				count += 1
			else:
				continue
		if count == len(a):
			b[5*k:5*(k+1)] = a
			break


print(b)
'''b = [0,0,0,0,0,0,0,0,0,0,0,0]
l = len(a)
for k in range(len(a)):
	start = max(0,k-1)
	end = min(l-1,k+1)
	while True:
		r = random.randint(start, end)
		print(r)
		if b[r] == 0:
			b[r] = a[k]
			break
	print(b)

print(b)'''