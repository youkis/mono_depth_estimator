import argparse
parser = argparse.ArgumentParser()
parser.add_argument('a', type=str)
parser.add_argument('b', type=str)
args = parser.parse_args()

import numpy as np
a=np.loadtxt(args.a)
b=np.loadtxt(args.b)
print('band:', np.unique(a).size)
print('band:', np.unique(b).size)
if(a.size!=b.size): print('warning: size is different')
indice = (a!=0) & (b!=0)
indice2 = (indice==False)
print('mae:',(abs(a-b)).sum()/a.size)
print('mae(non-zero):',(abs(a[indice]-b[indice])).sum()/indice.size)
print('mae(zero):',(abs(a[indice2]-b[indice2])).sum()/indice2.size)
print('differ:%d/%d (%d%%)' % ((a!=b).sum(), a.size, (a!=b).sum()/a.size*100.0))

