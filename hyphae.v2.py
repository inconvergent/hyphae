#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():

  from numpy import cos, sin, pi, arctan2, sqrt, square, int
  from numpy.random import random as rand
  import numpy as np
  import cairo,Image
  from time import time as time
  from operator import itemgetter

  any = np.any
  all = np.all

  PI    = pi
  PII   = 2*pi
  N     = 5000
  ONE   = 1./N
  BACK  = 1.
  FRONT = 0.
  MID   = 0.5
  ALPHA = 0.2
  OUT   = './img'

  R     = 9./N;
  R2    = 2*R;

  ZONES = 200

  #def stroke(x,y):
    #ctx.rectangle(x,y,ONE,ONE)
    #ctx.fill()
    #return
  #vstroke = np.vectorize(stroke)

  #def get_colors(f):
    #scale = 255.
    #im = Image.open(f)
    #w,h = im.size
    #rgbim = im.convert('RGB')
    #res = {}
    #for i in xrange(w):
      #for j in xrange(h):
        #r,g,b = rgbim.getpixel((i,j))
        #key = '{:03d}{:03d}{:03d}'\
          #.format(r,g,b)
        #res[key] = (r/scale,g/scale,b/scale)
    #res = [value for key,value in res.iteritems()]
    #return res
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
  X   = np.zeros(nmax,dtype=np.float)
  Y   = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)

  X[0]   = 0.5
  Y[0]   = 0.5
  THE[0] = 0.

  i = 1+int(0.5*ZONES) 
  j = 1+int(0.5*ZONES) 
  ij = i*ZONES+j
  Z[ij].append(0)

  num = 1
  itt = 0
  ti = time()
  drawn = -1

  ctx.set_source_rgba(FRONT,FRONT,FRONT)

  while True:
    itt += 1

    k    = int(rand()*num)
    the  = (PI*(0.5-rand()))+THE[k];
    x    = X[k] + sin(the)*R2;
    y    = Y[k] + cos(the)*R2;
    
    inds = near_zone_inds(x,y,Z)
    good = True
    if len(inds)>0:
      dd = square( X[inds]-x ) + \
           square( Y[inds]-y )

      sqrt(dd,dd)
      mask = dd<sqrt(2*R*R)
      good = not any(mask)
      
    if good: 
      X[num]   = x
      Y[num]   = y
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

#dx = X[k]-x
#dy = Y[k]-y
#a = arctan2(dy,dx)
#scales = rand(CRN)*R2
#xp = X[k] - scales*cos(a)
#yp = Y[k] - scales*sin(a)
#cr = (0.5+0.6*rand(CRN))*CR
#vcirc(xp,yp,cr)
