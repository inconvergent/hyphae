#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt, square, int, linspace
from numpy.random import random as rand
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from numpy.random import normal as norm


any = np.any
all = np.all

N = 1000
ZONES = N/20
ONE = 1./N
BACK = 1.
FRONT = 0.
MID = 0.5


filename = 'res/test'
DRAW_SKIP = 10000

#COLOR_FILENAME = 'color/dark_cyan_white_black.gif'
#COLOR_FILENAME = 'color/light_brown_mushrooms.gif'
#COLOR_FILENAME = 'color/dark_brown_mushrooms.gif'
COLOR_FILENAME = 'color/dark_green_leaf.gif'

RAD = 3*ONE;
R_RAND_SIZE = 10
CK_MAX = 15

LINE_NOISE = 1.
SEARCH_ANGLE = 0.3*pi
SOURCE_NUM = 3


ALPHA = 0.5
GRAINS = 3

print
print 'filename',filename
print 'N', N
print 'one', ONE


class Render(object):

  def __init__(self,n):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,n,n)
    ctx = cairo.Context(sur)
    ctx.scale(n,n)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def get_colors(self,f):
    
    import Image
    from random import shuffle

    scale = 1./255.
    im = Image.open(f)
    w,h = im.size
    rgbim = im.convert('RGB')
    res = []
    for i in xrange(0,w):
      for j in xrange(0,h):
        r,g,b = rgbim.getpixel((i,j))
        res.append((r*scale,g*scale,b*scale))

    shuffle(res)

    self.colors = res
    self.ncolors = len(res)

  def line(self,x1,y1,x2,y2):

    self.ctx.set_source_rgba(FRONT,FRONT,FRONT)
    self.ctx.set_line_width(ONE*2.)
    self.ctx.move_to(x1,y1)
    self.ctx.line_to(x2,y2)
    self.ctx.stroke()

  def sandpaint_line(self,x1,y1,x2,y2,r):

    dx = x1-x2
    dy = y1-y2
    a = arctan2(dy,dx)
    dots = 2*int(r*N)
    scales = linspace(0,r,dots)
    xp = x1 - scales*cos(a) + rand(dots)*ONE*LINE_NOISE
    yp = y1 - scales*sin(a) + rand(dots)*ONE*LINE_NOISE

    self.ctx.set_source_rgba(FRONT,FRONT,FRONT)

    for x,y in zip(xp,yp):
      self.ctx.rectangle(x,y,ONE,ONE) 
      self.ctx.fill()

  def sandpaint_color_line(self,x1,y1,x2,y2,r,k):

    dx = x1 - x2
    dy = y1 - y2
    a = arctan2(dy,dx)
    scales = rand(GRAINS)*r
    xp = x1 - scales*cos(a)
    yp = y1 - scales*sin(a)

    r,g,b = self.colors[k%self.ncolors]
    self.ctx.set_source_rgba(r,g,b,ALPHA)

    for x,y in zip(xp,yp):
      self.ctx.rectangle(x,y,ONE,ONE) 
      self.ctx.fill()

def near_zone_inds(x,y,Z,k):
  
  i = 1+int(x*ZONES) 
  j = 1+int(y*ZONES) 
  ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
       np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])

  it = itemgetter(*ij)
  its = it(Z)
  inds = np.array([b for a in its for b in a if not b==k])

  return inds

def get_z(x,y):

  i = 1+int(x*ZONES) 
  j = 1+int(y*ZONES) 
  z = i*ZONES+j
  return z

def get_relative_search_angle():

  a = norm()*SEARCH_ANGLE
  #a = (0.5-rand())*SEARCH_ANGLE
  
  return a


def main():

  render = Render(N)
  render.get_colors(COLOR_FILENAME)

  Z = [[] for i in xrange((ZONES+2)**2)]

  nmax = 2*1e7
  R = np.zeros(nmax,dtype=np.float)
  X = np.zeros(nmax,dtype=np.float)
  Y = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)

  C = np.zeros(nmax,dtype=np.int)

  ## number of nodes
  num = 0

  ## init
  for i in xrange(SOURCE_NUM):

    X[i] = rand()
    Y[i] = rand()
    THE[i] = rand()*pi*2.

    z = get_z(X[i],Y[i])
    Z[z].append(num)
    num += 1

  itt = 0
  ti = time()
  drawn = -1

  while True:
    try:
      itt += 1

      k = int(rand()*num)
      C[k] += 1

      if C[k] > CK_MAX:
        continue

      the = get_relative_search_angle()+THE[k]
      r = RAD  + rand()*ONE*R_RAND_SIZE
      x = X[k] + sin(the)*r
      y = Y[k] + cos(the)*r
      
      ## if we are on the edge the zone mapping will fail.
      ## retry. do not draw sandpaint_color_line
      try:
        inds = near_zone_inds(x,y,Z,k)
      except IndexError:
        continue
        ## re-raise instead for the process to crash and burn.
        #raise

      good = True
      if len(inds)>0:
        dd = square(X[inds]-x) + square(Y[inds]-y)

        sqrt(dd,dd)
        mask = dd*2 >= (R[inds] + r)
        good = mask.all()
        
      if good: 
        X[num] = x
        Y[num] = y
        R[num] = r
        THE[num] = the

        z = get_z(x,y) 

        ## populate the zone map
        Z[z].append(num)
        
        ## draw the things
        render.line(X[k],Y[k],x,y)
        #render.sandpaint_line(X[k],Y[k],x,y,r)

        num+=1

      else:

        pass

        ## failed to add edge. draw colored edge
        #render.sandpaint_color_line(X[k],Y[k],x,y,r,k)
        
      if not num % DRAW_SKIP and not num==drawn:
        render.sur.write_to_png('{:s}.{:d}.png'.format(filename,num))
        print itt, num, time()-ti
        ti = time()
        drawn = num

    except KeyboardInterrupt:
      break

  return

if __name__ == '__main__':
  if True:
    import pstats, cProfile
    OUT = 'profile'
    pfilename = 'profile.profile'
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

