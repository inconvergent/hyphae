#!/usr/bin/python
# -*- coding: utf-8 -*-


from numpy import cos, sin, pi, arctan2
from numpy.random import random as rand
import numpy as np
import cairo,Image
from time import time as time
from operator import itemgetter

PI    = pi
PII   = 2*pi
N     = 21000
BACK  = 1.
FRONT = 0.
ALPHA = 0.05
OUT   = './xx/hyphae.xx'

R     = 7./N;
R2    = 2*R;
RR9   = 2*R*R;
GRAINS= 20

ZONES = 210

def stroke(x,y):
  ctx.rectangle(x,y,1./N,1./N)
  ctx.fill()
  return
vstroke = np.vectorize(stroke)

def getColors(f):
  scale = 255.
  im = Image.open(f)
  w,h = im.size
  rgbim = im.convert('RGB')
  res = {}
  for i in xrange(w):
    for j in xrange(h):
      r,g,b = rgbim.getpixel((i,j))
      key = '{:03d}{:03d}{:03d}'\
        .format(r,g,b)
      res[key] = (r/scale,g/scale,b/scale)
  res = [value for key,value in res.iteritems()]
  return res
colors = getColors('../orbitals/resources/colors2.gif')
ncolors = len(colors)

def ctxInit():
  sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,N,N)
  ctx = cairo.Context(sur)
  ctx.scale(N,N)
  ctx.set_source_rgb(BACK,BACK,BACK)
  ctx.rectangle(0,0,1,1)
  ctx.fill()
  return sur,ctx
sur,ctx = ctxInit()

def nearZoneInds(x,y,Z):
  
  i = 1+int(x*ZONES) 
  j = 1+int(y*ZONES) 
  ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
       np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])
  it = itemgetter(*ij)
  its = it(Z)
  inds = np.array([b for a in its for b in a ])

  return inds


def run(num,X,Y,THE,Z):
  
  itt = 0
  ti = time()
  while True:
    itt += 1

    k = int(rand()*num)
    the = (PI*(0.5-rand()))+THE[k];
    x = X[k] + sin(the)*R2;
    y = Y[k] + cos(the)*R2;

    inds = nearZoneInds(x,y,Z)
    if len(inds)>0:
      """ old
      nx = X[0:num] - x
      ny = Y[0:num] - y
      good = not (nx**2 + ny**2 < RR9).any()
      """

      nx = X[inds] - x
      ny = Y[inds] - y
      good = not (nx**2 + ny**2 < RR9).any()
    
      
      if good: 
        X[num] = x
        Y[num] = y
        THE[num] = the

        i = 1+int(x*ZONES) 
        j = 1+int(y*ZONES) 
        ij = i*ZONES+j
        Z[ij].append(num)
        
        ctx.set_source_rgb(FRONT,FRONT,FRONT)
        ctx.move_to(x,y)
        ctx.line_to(X[k],Y[k])
        ctx.stroke()
        num+=1

      else:
        r,g,b = colors[k%ncolors]
        ctx.set_source_rgba(r,g,b,ALPHA)
        dx = X[k] - x
        dy = Y[k] - y
        a = arctan2(dy,dx)
        scales = rand(GRAINS)*R2
        xp = X[k] - scales*cos(a)
        yp = Y[k] - scales*sin(a)

        vstroke(xp,yp)

    if not itt % 100000:
      sur.write_to_png('{:s}.{:d}.png'.format(OUT,itt,num))
      print itt, num, time()-ti
      ti = time()

      
def main():

  ctx.set_line_width(2./N)

  num = 1
 
  Z = [[] for i in xrange((ZONES+2)**2)]

  nmax = 2*1e7
  X   = np.zeros(nmax,dtype=np.float)
  Y   = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)

  X[0] = 0.5
  Y[0] = 0.5
  THE[0] = 0.

  i = 1+int(0.5*ZONES) 
  j = 1+int(0.5*ZONES) 
  ij = i*ZONES+j
  Z[ij].append(0)

  run(num,X,Y,THE,Z)

  return

if __name__ == '__main__' : main()

