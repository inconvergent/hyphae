#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():

  from numpy import cos, sin, pi, arctan2, sqrt, square, int, linspace
  from numpy.random import random as rand
  from scipy.spatial import Delaunay as triag
  from scipy.spatial import Voronoi, voronoi_plot_2d, distance
  from scipy.sparse import coo_matrix
  import numpy as np
  import cairo,Image
  from time import time as time
  from operator import itemgetter

  import matplotlib.pyplot as plt

  any   = np.any
  all   = np.all
  hstack = np.hstack
  vstack = np.vstack
  colstack = np.column_stack
  cdist = distance.cdist
  zeros = np.zeros

  pii   = 2*pi
  N     = 800
  ONE   = 1./N
  BACK  = 1.
  FRONT = 0.
  MID   = 0.5
  ALPHA = 0.2
  OUT   = './img'

  RAD   = 3./N;

  ZONES = 80

  print N / float(ZONES)

  def ctx_init():
    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,N,N)
    ctx = cairo.Context(sur)
    ctx.scale(N,N)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()
    return sur,ctx

  sur,ctx = ctx_init()
  ctx.set_source_rgba(0,0,1,0.7)
  ctx.set_line_width(ONE)
  
  def stroke(x,y): 
    ctx.rectangle(x,y,ONE,ONE) 
    ctx.fill() 
  vstroke = np.vectorize(stroke) 

  def circ(x,y,r): 
    ctx.arc(x,y,r,0,pii) 
    ctx.fill() 
  vcirc = np.vectorize(circ) 

    #points  (ndarray of double, shape (npoints, ndim)) Coordinates of input
      #points.

    #vertices  (ndarray of double, shape (nvertices, ndim)) Coordinates of the
      #Voronoi vertices.

    #ridge_points  (ndarray of ints, shape (nridges, 2)) Indices of the points
      #between which each Voronoi ridge lies.

    #ridge_vertices  (list of list of ints, shape (nridges, *)) Indices of the
      #Voronoi vertices forming each Voronoi ridge.

    #regions (list of list of ints, shape (nregions, *)) Indices of the Voronoi
      #vertices forming each Voronoi region. -1 indicates vertex outside the
      #Voronoi diagram.

    #point_region  (list of ints, shape (npoints)) Index of the Voronoi region
      #for each input point. If qhull option “Qc” was not specified, the list will
      #contain -1 for points that are not associated with a Voronoi region.

  def draw_tessellation(voronoi):

    #R = cdist( *[ voronoi.vertices ]*2 ) 

    for region in voronoi.point_region:
      inds = voronoi.regions[region]

    #for inds in voronoi.regions:

      if inds and not -1 in inds:

        xy = voronoi.vertices[inds[0],:]
        ctx.move_to(xy[0],xy[1])
        for ind in inds[1:]:
          xy = voronoi.vertices[ind,:]
          ctx.line_to(xy[0],xy[1])

        ctx.close_path()

        ctx.set_source_rgba(0,0,0,0.5)
        ctx.stroke()

  def neighborhood_matrix(voronoi):
    
    ii = []
    jj = []
    for i,indsi in enumerate(voronoi.regions):
      if indsi and not -1 in indsi:
        a = set(indsi)
        for j,indsj in enumerate(voronoi.regions):
          if indsj and not -1 in indsj:
            b = set(indsj)
            
            if len(a.intersection(b))>1:

              ## they are neighbors
              ii.append(i)
              jj.append(j)
    
    n = len(voronoi.regions)
    F = coo_matrix(([1]*len(ii),(ii,jj)), shape=(n,n)).tocsr()

    return F

  def draw_friends(voronoi, F):
    
    n = len(voronoi.regions)
    for r in xrange(n):
      row = F.getrow(r)

      if row.getnnz().sum()>5:

        inds = voronoi.regions[r]
        xy = voronoi.vertices[inds[0],:]
        ctx.move_to(xy[0],xy[1])
        for ind in inds[1:]:
          xy = voronoi.vertices[ind,:]
          ctx.line_to(xy[0],xy[1])

        ctx.close_path()

        ctx.set_source_rgba(0,0,0,ALPHA)
        ctx.fill()

  def near_zone_inds(x,y,Z):
    
    i = 1+int(x*ZONES) 
    j = 1+int(y*ZONES) 
    ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
         np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])

    it = itemgetter(*ij)
    its = it(Z)
    inds = np.array([b for a in its for b in a ])

    return inds
  
  Z = [[] for i in xrange((ZONES+2)**2)]

  nmax = 2*1e7
  R = np.zeros(nmax,dtype=np.float)
  X = np.zeros(nmax,dtype=np.float)
  Y = np.zeros(nmax,dtype=np.float)
  THE = np.zeros(nmax,dtype=np.float)

  X[0]   = 0.5
  Y[0]   = 0.5
  THE[0] = 0

  i = 1+int(0.5*ZONES) 
  j = 1+int(0.5*ZONES) 
  ij = i*ZONES+j
  Z[ij].append(0)

  num = 1
  itt = 0
  ti = time()
  drawn = -1

  for itt in xrange(1000000):

    k = int(sqrt(rand())*num)
    the = ( pi*( 0.5-rand() ) )+THE[k];
    r = sqrt((1 + rand()*40))*ONE
    x = X[k] + cos(the)*(r+R[k])
    y = Y[k] + sin(the)*(r+R[k])
    
    inds = [i for i in near_zone_inds(x,y,Z) if i!=k]
    good = True
    if len(inds)>0:
      dd = sqrt( square( X[inds]-x ) + \
                 square( Y[inds]-y ) )

      mask = (dd > (R[inds] + r))
      good = mask.all()

    if good: 
      X[num] = x
      Y[num] = y
      R[num] = r
      THE[num] = the

      i = 1+int(x*ZONES) 
      j = 1+int(y*ZONES) 
      Z[i*ZONES+j].append(num)

      num+=1

      #ctx.move_to(X[k],Y[k])
      #ctx.line_to(x,y)
      #ctx.stroke()
    
      if num > 500:
        break
      
  XY = colstack(( X[:num],Y[:num] ))
  voronoi = Voronoi(XY)

  ctx.set_source_rgb(1,0,0)
  vcirc( X[:num],Y[:num],ONE*1.5 )

  F = neighborhood_matrix(voronoi)

  food_index = ( rand(5)*num ).astype(np.int)

  resources = zeros((num,),dtype=np.float) + 1.
  resources_new = zeros(resources.shape,resources.dtype)

  resources[food_index] += 1.

  for itt in xrange(100):

    resources *= 0.98
    resources[food_index] += 1.

    for r in xrange(num):

      resources_new[:] = 0.

      row = F.getrow(r)
      nz = row.nonzero()[1]
      print nz
      if nz.sum()>1:
        resources_new[nz] = resources[nz].mean()

      resources[:] = resources_new[:]

  print resources, resources_new


  ## done
  sur.write_to_png('{:s}.{:d}.png'.format(OUT,num))
  print itt, num, time()-ti
  ti = time()

  return

if __name__ == '__main__':
  main()

