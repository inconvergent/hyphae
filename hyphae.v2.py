#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():

  from numpy import cos, sin, pi, arctan2, sqrt, square, int
  from numpy.random import random as rand
  import numpy as np
  import cairo,Image
  from time import time as time
  from operator import itemgetter

  any   = np.any
  all   = np.all

  PI    = pi
  PII   = 2*pi
  N     = 4000
  ONE   = 1./N
  BACK  = 1.
  FRONT = 0.
  MID   = 0.5
  ALPHA = 0.2
  OUT   = './img'

  RAD   = 3./N;

  ZONES = 400

  OBJECTS = 5

  colors = [(0,0,0)]
  ncolors = 1

  def ctx_init():
    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,N,N)
    ctx = cairo.Context(sur)
    ctx.scale(N,N)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()
    return sur,ctx
  sur,ctx = ctx_init()

  def near_zone_inds(x,y,Z):
    
    i  = 1+int(x*ZONES) 
    j  = 1+int(y*ZONES) 
    ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
         np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])

    it   = itemgetter(*ij)
    its  = it(Z)
    inds = np.array([b for a in its for b in a ])

    return inds
  
  ctx.set_line_width(2./N)

  Z = [[] for i in xrange((ZONES+2)**2)]

  nmax = 2*1e7
  R   = np.zeros(nmax,dtype=np.float)
  X   = np.zeros(nmax,dtype=np.float)
  Y   = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)


  X[0]   = 0.5
  Y[0]   = 0.5
  THE[0] = RAD

  i = 1+int(0.5*ZONES) 
  j = 1+int(0.5*ZONES) 
  ij = i*ZONES+j
  Z[ij].append(0)

  OR = np.zeros(OBJECTS,dtype=np.float)
  OX = np.zeros(OBJECTS,dtype=np.float)
  OY = np.zeros(OBJECTS,dtype=np.float)

  for i in xrange(OBJECTS):
    OR[i] = ONE*200
    OX[i] = rand()
    OY[i] = rand()

  num = 1
  itt = 0
  ti = time()
  drawn = -1

  ctx.set_source_rgba(FRONT,FRONT,FRONT)

  while True:
    itt += 1

    k    = int(sqrt(rand())*num)
    the  = ( PI*( 0.5-rand() ) )+THE[k];
    r    = RAD  + rand()*ONE*4.
    x    = X[k] + sin(the)*r;
    y    = Y[k] + cos(the)*r;

    #dd = square( OX-x ) + \
         #square( OY-y )
    #sqrt(dd,dd)
    #mask = dd*2 < OR 350
    #blocked_by_object = mask.any()

    #if blocked_by_object:
      #continue
    
    inds = near_zone_inds(x,y,Z)
    good = True
    if len(inds)>0:
      dd = square( X[inds]-x ) + \
           square( Y[inds]-y )

      sqrt(dd,dd)
      mask = dd*2 >= R[inds] + r
      good = mask.all()

      
    if good: 
      X[num]   = x
      Y[num]   = y
      R[num]   = r
      THE[num] = the

      i = 1+int(x*ZONES) 
      j = 1+int(y*ZONES) 
      Z[i*ZONES+j].append(num)
      
      ctx.move_to(X[k],Y[k])
      ctx.line_to(x,y)
      ctx.stroke()

      num+=1
      
    if not num % 1000 and not num==drawn:
      sur.write_to_png('{:s}.{:d}.png'.format(OUT,num))
      print itt, num, time()-ti
      ti = time()
      drawn = num

  return

if __name__ == '__main__':
  main()

