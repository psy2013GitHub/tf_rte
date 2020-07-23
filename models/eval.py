#!/usr/bin/python -w
# eval.py: evaluate accuracy of rte task
# usage:     python eval.py  file
# options:   file:

import sys
n, acc = 0, 0
with open(sys.argv[1], 'r') as fid:
   for line in fid:
      line = line.strip()
      if not line:
         continue
      n += 1
      segs = line.split()
      acc += segs[1] == segs[2]
print(acc/n)


