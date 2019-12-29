import numpy as np
import matplotlib.pyplot as plt
def square(L, a, x_0, y_0):
	# All values in pixels
	# L = size of imaging square 
	# a = size of square 
	# x_0, y_0 = coordinates of center of square
	x = np.zeros([L,L], dtype = np.complex128)
	for i in range(L):
		for j in range(L):
			if abs(i-x_0) <= a/2 and abs(j-x_0) <= a/2:
				x[j,i] = 1.0

	return x

def rectangle(L, a, b, x_0, y_0):
	# All values in pixels
	# L = size of imaging square 
	# a, b = rectangle dimensions 
	# x_0, y_0 = coordinates of center of rectangle
	x = np.zeros([L,L], dtype = np.complex128)
	for i in range(L):
		for j in range(L):
			if abs(i-x_0) <= a/2 and abs(j-y_0) <= b/2:
				x[j,i] = 1.0

	return x

def circle(L, a, x_0, y_0):
	# All values in pixels
	# L = size of imaging square 
	# a = radius of circle 
	# x_0, y_0 = coordinates of center of square
	x = np.zeros([L,L], dtype = np.complex128)
	for i in range(L):
		for j in range(L):
			if abs(i-x_0)**2 + abs(j-y_0)**2 <= a**2:
				x[j,i] = 1.0

	return x


def circle_absolute(d, L, a, x_0, y_0):
	# All values in pixels
	# d = Size of imaging domain (in m)
	# L = size of imaging square
	# a, x_0, y_0 = radius of circle ,coordinates of center of square
	# All in (in m) 
	
	x = np.zeros([L,L], dtype = np.complex128)
	n = d/L
	for i in range(L):
		for j in range(L):
			i_m, j_m =  (i+0.5)*n, (j+0.5)*n
			if abs(i_m-x_0)**2 + abs(j_m-y_0)**2 <= a**2:
				x[j,i] = 1.0

	return x

def austria_absolute(d,L):
	x1 = circle_absolute(d, L, 0.6, 1, 1.2)
	x2 = circle_absolute(d, L, 0.3, 1, 1.2)
	x3 = circle_absolute(d, L, 0.2, 0.7, 0.4)
	x4 = circle_absolute(d, L, 0.2, 1.3, 0.4)

	return (x1 - x2) + x3 + x4




def austria_multicontrast(L, c1, c2, c3):
	x1 = circle(L, L*0.3,  L*0.5, L*0.6)
	x2 = circle(L, L*0.15, L*0.5, L*0.6)
	x3 = circle(L, L*0.1,  L*0.35, L*0.2)
	x4 = circle(L, L*0.1,  L*0.65, L*0.2)

	return c1*(x1 - x2) + c2*x3 + c3*x4

def lossy_profile(L):
	# Obtained from NIE paper, Figure 6(a)
	# Title: A Fast Integral Equation-Based Method for Solving Electromagnetic Inverse Scattering Problems With Inhomogeneous Background
	x1 = circle(L, L*0.15, L*0.3, L*0.2)
	x2 = circle(L, L*0.15, L*0.7, L*0.2)
	x3 = rectangle(L, L*0.5, L*0.3, L*0.5, L*0.65)
	x4 = rectangle(L, L*0.8, L*0.5, L*0.5, L*0.65)
	c1 = 2.0
	c2 = 2.0
	c3 = 1.5 - 1j*0.5
	c4 = 0.5 - 1j*0.2

	return c1*x1 + c2*x2 + (c3 - c4)*x3 +  c4*x4