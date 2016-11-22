#!/usr/bin/env python3.5
from pylab import * 
import numpy as np 
import os, sys
import pretty as pr 
import scipy.optimize as opt

def curl(v):

	'''
	compute the curl of a 2d array of vectors

	Parameters:
		v 		array-like, shape = (n, n, 3)
	'''

	s = np.zeros_like(v)

	s[:,:,0] = np.gradient(v[:,:,2], axis=1)
	s[:,:,1] = - np.gradient(v[:,:,2], axis=0)
	s[:,:,2] = np.gradient(v[:,:,1], axis=0) - np.gradient(v[:,:,0], axis=1)   

	return s

def bpara_cross_bpara(v):

	s = np.zeros_like(v)

	dbz_by_dy = np.gradient(v[:,:,2], axis=1)
	dbz_by_dx = np.gradient(v[:,:,2], axis=0)

	s[:,:,0] = v[:,:,2] * dbz_by_dx
	s[:,:,1] = v[:,:,2] * dbz_by_dy

	return s

def bperp_cross_bpara(v):

	s = np.zeros_like(v)

	dbz_by_dy = np.gradient(v[:,:,2], axis=1)
	dbz_by_dx = np.gradient(v[:,:,2], axis=0)

	s[:,:,2] = v[:,:,0] * dbz_by_dx - v[:,:,1] * dbz_by_dy

	return s


def cross(u, v):

	'''
	compute the curl of a 2d array of vectors

	Parameters:
		v 		array-like, shape = (n, n, 3)
	'''

	s = np.zeros_like(u)

	s[:,:,0] = v[:,:,2] * u[:,:,1] - v[:,:,1] * u[:,:,2]
	s[:,:,1] = v[:,:,0] * u[:,:,2] - v[:,:,2] * u[:,:,0]
	s[:,:,2] = v[:,:,1] * u[:,:,0] - v[:,:,0] * u[:,:,1]

	return s


times = np.arange(1,15).astype(int)
nt = len(times)
strength1 = np.zeros(nt)
strength2 = np.zeros(nt)
bb = np.zeros(nt)

for j in times:
	fname = "output_Bpara_400_j1.in"
	p = "xy"

	v = np.zeros( (200, 200, 3) )

	v[:,:,0] = np.loadtxt( "{}/agl{}/{}_bx.agl".format(fname, j, p) )
	v[:,:,1] = np.loadtxt( "{}/agl{}/{}_by.agl".format(fname, j, p) )
	v[:,:,2] = np.loadtxt( "{}/agl{}/{}_bz.agl".format(fname, j, p) )
	b = np.loadtxt( "{}/agl{}/{}_B.agl".format(fname, j, p) )
	bb[j-1] = np.mean( b**2 )

	#curl_v = curl(v)
	#cross_term = cross(v, curl_v)

	term1 = bperp_cross_bpara(v)
	term2 = bpara_cross_bpara(v)

	strength1[j-1] = np.mean(np.sqrt(term1[:,:,0]**2 + term1[:,:,1]**2 + term1[:,:,2]**2) )
	strength2[j-1] = np.mean(np.sqrt(term2[:,:,0]**2 + term2[:,:,1]**2 + term2[:,:,2]**2) )

	#print (j, np.mean(strength1), np.mean(strength2))

pr.set_pretty()
plot(times, bb, c = "k")
plot(times, strength1, label = r"$B_{\perp} \times (\nabla \times B_{\parallel})$")
plot(times, strength2, label = r"$B_{\parallel  } \times (\nabla \times B_{\parallel})$")

hlines([1], 0,20, linestyle="--")
semilogy()
legend()
savefig("curl_{}.png".format(fname))

