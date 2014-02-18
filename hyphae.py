#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, int, linspace, any, all
from numpy.random import random as random
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from numpy.random import normal as normal


NMAX = 2*1e7 # maxmimum number of nodes
SIZE = 20000
ONE = 1./SIZE

RAD = 20.*ONE # 

ZONEWIDTH = 2.*(RAD/ONE)

ZONES = int(SIZE/ZONEWIDTH)
#ZONES = SIZE/20

BACK = 1.
FRONT = 0.
MID = 0.5

X_MIN = 0.+10.*ONE # border
Y_MIN = 0.+10.*ONE #
X_MAX = 1.-10.*ONE #
Y_MAX = 1.-10.*ONE #

filename = 'aa'
DRAW_SKIP = 5000 # write image this often

#COLOR_FILENAME = 'color/dark_cyan_white_black.gif'
#COLOR_FILENAME = 'color/light_brown_mushrooms.gif'
#COLOR_FILENAME = 'color/dark_brown_mushrooms.gif'
#COLOR_FILENAME = 'color/dark_green_leaf.gif'

RAD_SCALE = 0.95
R_RAND_SIZE = 6
CK_MAX = 30 # max number of allowed branch attempts from a node

CIRCLE_RADIUS = 0.4

SEARCH_ANGLE = 0.22*pi
SOURCE_NUM = 20

ALPHA = 0.09
GRAINS = 10


print
print 'filename',filename
print 'SIZE', SIZE
print 'ZONEWIDTH', ZONEWIDTH
print 'RAD', RAD
print 'ZONES', ZONES
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

    self.colors = ((0,0,0))
    self.ncolors = 1

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

    #self.ctx.set_line_width(ONE*2.)
    self.ctx.move_to(x1,y1)
    self.ctx.line_to(x2,y2)
    self.ctx.stroke()

  def circle(self,x,y,r):

    self.ctx.arc(x,y,r,0,pi*2.)
    self.ctx.stroke()

  def circle_fill(self,x,y,r):

    self.ctx.arc(x,y,r,0,pi*2.)
    self.ctx.fill()

  def circles(self,x1,y1,x2,y2,r):

    dx = x1-x2
    dy = y1-y2
    dd = sqrt(dx*dx+dy*dy)

    n = int(dd/ONE)
    n = n if n>6 else 6

    a = arctan2(dy,dx)

    #scale = random(n)*dd
    scale = linspace(0,dd,n)

    xp = x1-scale*cos(a)
    yp = y1-scale*sin(a)

    ## random radius?
    for x,y in zip(xp,yp):
      self.ctx.arc(x,y,r,0,pi*2.) 
      self.ctx.fill()

  def sandpaint_line(self,x1,y1,x2,y2,r):

    dx = x1-x2
    dy = y1-y2
    a = arctan2(dy,dx)
    dots = 2*int(r*SIZE)
    scales = linspace(0,r,dots)
    xp = x1 - scales*cos(a) + random(dots)*ONE*LINE_NOISE
    yp = y1 - scales*sin(a) + random(dots)*ONE*LINE_NOISE

    self.ctx.set_source_rgba(FRONT,FRONT,FRONT)

    for x,y in zip(xp,yp):
      self.ctx.rectangle(x,y,ONE,ONE) 
      self.ctx.fill()

  def sandpaint_color_line(self,x1,y1,x2,y2,k):

    dx = x1 - x2
    dy = y1 - y2
    dd = sqrt(dx*dx+dy*dy)
    a = arctan2(dy,dx)
    scales = random(GRAINS)*dd
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
  #a = (0.5-random())*SEARCH_ANGLE
  
  return a


def main():

  render = Render(SIZE)
  #render.get_colors(COLOR_FILENAME)

  Z = [[] for i in xrange((ZONES+2)**2)]

  R = np.zeros(NMAX,'float')
  X = np.zeros(NMAX,'float')
  Y = np.zeros(NMAX,'float')
  THE = np.zeros(NMAX,'float')
  P = np.zeros(NMAX,'int')
  C = np.zeros(NMAX,'int')
  D = np.zeros(NMAX,'int')-1

  ## number of nodes
  i = 0

  ## initialize source nodes
  while i<SOURCE_NUM:

    ## in circle
    #x = random()
    #y = random()
    #if sqrt(square(x-0.5)+square(y-0.5))<0.3:
      #X[i] = x
      #Y[i] = y
      #R[i] = (RAD + 0.2*RAD*(1.-2.*random()))
      #P[i] = -1
    #else:

      ### try again
      #continue

    ## on canvas
    X[i] = random()
    Y[i] = random()

    ## on circle
    #gamma = i*2.*pi/float(SOURCE_NUM)
    #X[i] = 0.5 + cos(gamma)*0.1
    #Y[i] = 0.5 + sin(gamma)*0.1

    THE[i] = random()*pi*2.
    P[i] = -1 # no parent
    R[i] = RAD
    z = get_z(X[i],Y[i])
    Z[z].append(i)
    i += 1

  num = i

  itt = 0
  ti = time()
  drawn = -1

  while True:

    try:

      itt += 1
      if not itt%1000:
        print itt, num

      added_new = False

      k = int(random()*num)
      C[k] += 1

      if C[k]>CK_MAX:

        ## node is dead
        continue

      #r = RAD + random()*ONE*R_RAND_SIZE
      r = R[k]*RAD_SCALE if D[k]>-1 else R[k]

      if r<ONE*0.5:

        ## node dies
        C[k] = CK_MAX+1
        continue

      #sa = normal()*SEARCH_ANGLE
      sa = normal()*(1.-r/(RAD+ONE))*pi
      the = sa+THE[k]

      x = X[k] + sin(the)*(r+R[k])
      y = Y[k] + cos(the)*(r+R[k])

      # stop nodes at edge of canvas
      if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:

        ## node is outside canvas
        continue

      ## stop nodes at edge of circle
      ## remember to set initial node inside circle.
      circle_rad = sqrt(square(x-0.5)+square(y-0.5))
      if circle_rad>CIRCLE_RADIUS:

        ## node is outside circle
        continue
      
      try:

        inds = near_zone_inds(x,y,Z,k)
      except IndexError:

        ## node is outside zonemapped area
        continue

      good = True
      if len(inds)>0:
        dd = square(X[inds]-x) + square(Y[inds]-y)

        sqrt(dd,dd)
        mask = dd > R[inds]+r
        good = mask.all()
        
      if good: 
        X[num] = x
        Y[num] = y
        R[num] = r
        THE[num] = the
        P[num] = k

        ## set first descendant if node has no descendants
        if D[k]<0:
          D[k] = num

        z = get_z(x,y) 

        Z[z].append(num)

        render.ctx.set_source_rgb(FRONT,FRONT,FRONT)
        render.circles(X[k],Y[k],x,y,r*0.6)


        ### render node radii
        #render.ctx.set_line_width(ONE)
        #render.ctx.set_source_rgba(1,0,0,1)
        #render.circle(x,y,r)

        num += 1
        added_new = True

        if not num % DRAW_SKIP and added_new:
          render.sur.write_to_png('{:s}.{:d}.png'.format(filename,num))
          print itt, num, time()-ti
          ti = time()
          drawn = num

      else:
        pass

        #render.sandpaint_color_line(X[k],Y[k],x,y,k)

    except KeyboardInterrupt:
      break

  return

if __name__ == '__main__':
  if False:
    import pstats, cProfile
    OUT = 'profile'
    pfilename = 'profile.profile'
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

