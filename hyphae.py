#!/usr/bin/python
# -*- coding: utf-8 -*-


from numpy import cos, sin, pi
from numpy.random import random as rand
import numpy as np
import cairo
from time import time as time

PI    = pi
PII   = 2*pi
N     = 5000
BACK  = 1.
FRONT = 0.
ALPHA = 1.
OUT   = 'hyp.a.0'

R     = 7./N;
R2    = 2*R;
RR9   = 2*R*R;

def ctxInit():
  sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,N,N)
  ctx = cairo.Context(sur)
  ctx.scale(N,N)
  ctx.set_source_rgb(BACK,BACK,BACK)
  ctx.rectangle(0,0,1,1)
  ctx.fill()
  return sur,ctx


def run(num,X,Y,THE,ctx,sur):
  
  itt = 0
  ti = time()
  while True:
    itt += 1

    k = int((0.5+0.5*rand())*num)
    the = (PI*(0.5-rand()))+THE[k];
    x = X[k] + sin(the)*R2;
    y = Y[k] + cos(the)*R2;

    nx = X[0:num] - x
    ny = Y[0:num] - y
    good = not (nx**2 + ny**2 < RR9).any()

    if good: 
      X[num] = x
      Y[num] = y
      THE[num] = the
      ctx.move_to(x,y)
      ctx.line_to(X[k],Y[k])
      ctx.stroke()
      num+=1

    if not itt % 10000:
      sur.write_to_png('{:s}.{:d}.{:d}.png'.format(OUT,itt,num))
      print itt, num, time()-ti
      ti = time()

      
def main():

  sur,ctx = ctxInit()
  ctx.set_source_rgba(FRONT,FRONT,FRONT,ALPHA)
  ctx.set_line_width(2./N)

  num = 1
  
  nmax = 1e7
  X   = np.zeros(nmax,dtype=np.float)
  Y   = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)

  X[0] = 0.5
  Y[0] = 0.5
  THE[0] = 0.5

  run(num,X,Y,THE,ctx,sur)

  return

if __name__ == '__main__' : main()

