import numpy as np
import cv2
import scipy.special as sc
import numpy.linalg as linalg
from scipy import fftpack

def construct_g_D(pos_D, k, n):
	# pos_D : N X 2 vector containing position of each grid
	# n : size of each grid
	# k : wavevector
	a = (n**2/np.pi)**0.5
	N = np.shape(pos_D)[0]
	L = np.int32(N**0.5)

	pos_D_x, pos_D_y = np.reshape(pos_D[:,0], [L,L]), np.reshape(pos_D[:,1], [L,L])
	g_D = np.zeros([2*L-1,2*L-1],dtype=np.complex128)
	g_D[0,0] = -1j*0.5*(np.pi*k*a*sc.hankel2(1,k*a) - 2*1j)
	for i in range(L):
		for j in range(1,L):			
			rho_ij = np.abs(pos_D_x[i,j]-pos_D_x[0,0])**2 + np.abs(pos_D_y[i,j]-pos_D_y[0,0])**2 
			rho_ij = np.float32(rho_ij**0.5)
			g_D[i,j] = -1j*0.5*np.pi*k*a*sc.j1(k*a)*sc.hankel2(0,k*rho_ij)

	for i in range(1,L):
		rho_ij = np.abs(pos_D_x[i,0]-pos_D_x[0,0])**2 + np.abs(pos_D_y[i,0]-pos_D_y[0,0])**2 
		rho_ij = np.float32(rho_ij**0.5)
		g_D[i,0] = -1j*0.5*np.pi*k*a*sc.j1(k*a)*sc.hankel2(0,k*rho_ij)

	for i in range(L,2*L-1):
		for j in range(L, 2*L-1):
			g_D[i,j] = g_D[abs(i+1-2*L),abs(j+1-2*L)]
	for i in range(L, 2*L-1):
		for j in range(L):
			g_D[i,j] = g_D[abs(i+1-2*L),j]
	for i in range(L):
		for j in range(L, 2*L-1):
			g_D[i,j] = g_D[i,abs(j+1-2*L)]
	g_D_fft = fftpack.fftn(g_D)
	g_D_fft_conj = fftpack.fftn(g_D.conj())
	return g_D, g_D_fft, g_D_fft_conj

def G_D_into_x(g_D_fft, w):
	N, V = np.shape(w)[0], np.shape(w)[1]
	L = np.int32(N**0.5)
	G_D_x = np.zeros([N,V],dtype=np.complex128)
	for v in range(V):
		w_im = np.zeros([2*L-1,2*L-1],dtype=np.complex128)
		w_im[0:L,0:L] = np.reshape(w[:,v], [L,L])
		fft_w = fftpack.fftn(w_im)
		fft_GDw = fft_w*g_D_fft
		y = fftpack.ifftn(fft_GDw)
		G_D_x[:,v] = np.reshape(y[:L,:L],[N])

	return G_D_x

def cg_fft_forward_problem(x, G_S, g_D_fft, e, tol, d0, max_iter):
	power10 = round(np.log10(1/tol), 0)
	tol_10power = np.power(10,power10)

	N = np.shape(x)[0] # Number of grids
	M = np.shape(G_S)[0] # Number of receivers
	V = np.shape(e)[1]   # Number of views
	# Scattered field given by the relation
	# s_v = G_S*X*L_X*e_v
	# where L_X = (I - G_D*X)^(-1)
	
	s = np.empty([M, V],dtype=np.complex128)
	X = np.diag(x[:,0])
	# Inverse calculation 1
	#L_X = linalg.inv(np.eye(N)- np.matmul(G_D,X))
	#d = np.matmul(L_X,e)		
 	#Inverse calculation 2 Solving A d = b
	d = np.zeros([N,V],dtype=np.complex128) # The internal field
	w = np.zeros([N,V],dtype=np.complex128) # The internal field

	for v in range(V):
		b = np.reshape(e[:,v],[N,1]) 
		d_old = np.reshape(d0[:,v],[N,1])  #Setting initial internal field to d0
		r0 = b - (d_old - G_D_into_x(g_D_fft,d_old*x))
		q = r0
		p = r0
		r_old = r0
		count = 0
		tol_cal = np.linalg.norm(r0,'fro')/np.linalg.norm(b,'fro')
		# print('Initial tol: %0.3fe-%d for %dth illumination'%(tol_cal*tol_10power,power10,v+1))
		d_new = d_old
		while count < max_iter and tol_cal > tol: # or convg = true #CGS
			Aq = q - G_D_into_x(g_D_fft,q*x)
			a = np.sum(r_old * r0.conj()) / np.sum(Aq * r0.conj())
			u = p - a*Aq
			d_new = d_old + a*(p+u)
			A_pu = p + u - G_D_into_x(g_D_fft,(p + u)*x)
			r_new = r_old - a*A_pu
			beta = np.sum(r_new * r0.conj())/ np.sum(r_old * r0.conj())
			p = r_new + beta * u
			q = p + beta*(u + beta*q)

			r_old = r_new
			d_old = d_new
			count = count + 1
			temp = b - (d_new - G_D_into_x(g_D_fft,d_new*x)) 
			tol_cal = np.linalg.norm(temp,'fro')/np.linalg.norm(b,'fro')
			# if np.mod(count, 100) == 0:
				# print('tol: %.3fe-%d iterations: %d for %dth illumination'%(tol_cal*tol_10power,power10,count,v+1))
		print('Tolerance = %0.3fe-%d at iteration = %d for %dth illumination'%(tol_cal*tol_10power,power10,count,v+1))
		if tol_cal > tol:
			print('Convergence failed. Run forward solver again.')
		d[:,v] = np.reshape(d_new,N)
		w[:,v] = np.reshape(x*d_new,N)
		s[:,v] = np.reshape(np.matmul(G_S,np.reshape(x*d_new,[N,1])),M)
	return s, w
