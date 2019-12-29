import numpy as np
import scipy.linalg as la
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
def ADMM_TV(A, b, lambda_D):
	N = np.shape(A)[1]
	L = np.int32(N**0.5)
	# L is the size of image
	D_X, D_Y = get_TV_mat(L)
	D = np.concatenate((D_X, D_Y))
	print(np.shape(D))
	
	max_iter = 50
	rho = 1e-2

	z = np.zeros([2*N,1])
	u = np.zeros([2*N,1])
	L,U = la.lu(np.matmul(A.T, A) + rho*np.matmul(D.T,D),permute_l=True)
	kappa = lambda_D/rho
	for iteration in range(max_iter):
		x_temp = np.linalg.solve( L, np.matmul(A.T, b) + rho*np.matmul(D.T, z-u) )
		x = np.linalg.solve(U, x_temp)
		z_temp = np.matmul(D,x) + u
		z = ((z_temp - kappa) >=0)*(z_temp - kappa) - ((-z_temp-kappa) >= 0)*(-z_temp-kappa)

		u = u + np.matmul(D, x) - z
	return x

# A and b all complex
# This version of ADMM_TV just breaks it down into real and imaginary subparts
def ADMM_TV_complex(A, b, lambda_D, x_0):
	N = np.shape(A)[1]
	# L is the size of image
	A_split_R = np.concatenate((np.real(A),-np.imag(A)),axis=1)
	A_split_I = np.concatenate((np.imag(A), np.real(A)),axis=1)
	A = np.concatenate((A_split_R,A_split_I),axis=0)
	b = np.concatenate((np.real(b),np.imag(b)),axis=0)

	L = np.int32((N)**0.5)
	D_X, D_Y = get_TV_mat(L)
	D_R = np.concatenate((D_X, D_Y, np.zeros([2*N,N])))
	D_I = np.concatenate((np.zeros([2*N,N]),D_X, D_Y))
	D = np.concatenate((D_R,D_I),axis=1)
	# D = np.eye(2*N)
	# x = np.concatenate((np.real(x),np.imag(x)))
	# cost_true_solution = np.linalg.norm(np.matmul(A,x) - b,2)**2 + lambda_D*np.linalg.norm(np.matmul(D,x),1)
	# print('Cost Function at true solution: %0.5f'%(cost_true_solution))
	
	x = np.concatenate((np.real(x_0),np.imag(x_0)))
	z = np.matmul(D, x)
	u = np.zeros([4*N,1])

	max_iter = 50
	rho = 0.01	
	verbose = True
	
	kappa = lambda_D/rho
	cost = np.inf
	A_T_A = np.matmul(A.T, A)
	D_T_A = np.matmul(D.T,D)
	x_pinv_A = A_T_A + rho*D_T_A
	[L,U] = la.lu(x_pinv_A,permute_l=True)
	for iteration in range(max_iter):
		x_temp = np.linalg.solve(L, np.matmul(A.T, b) + rho*np.matmul(D.T, z-u) )
		x = np.linalg.solve(U, x_temp)
		z_temp = np.matmul(D,x) + u
		z = (z_temp >= kappa)*(z_temp - kappa) + (z_temp <= -kappa)*(z_temp+kappa)
		u = u + (np.matmul(D, x) - z)

		residue, regularizer = np.linalg.norm(np.matmul(A,x)-b), np.linalg.norm(np.matmul(D,x),1)
		prev_cost = cost
		cost = residue**2 + lambda_D*regularizer		
		if prev_cost < 0.99*cost:
			break		
		if verbose:
			print('Iteration:  ',iteration)
			print('||Ax-b||_2: %0.3f, ||Dx||_1: %0.3f'%(residue**2, regularizer))
			print('Cost Function: %0.3f'%(cost))

	return x[:N,:] + 1j*x[N:,:]

def get_TV_mat(L):
	N = L*L
	D_X = np.zeros([N,N], dtype=np.float32)
	D_Y = np.zeros([N,N], dtype=np.float32)
	count = 0
	for i in range(L):
		for j in range(L):
			im = np.zeros([L,L],dtype=np.float32)
			im[i,j] = 1.0

			im_diff_x, im_diff_y = im - np.roll(im,-1, axis = 0), im - np.roll(im,-1, axis = 1)
			D_X[:,count] = np.reshape(im_diff_x,[N])
			D_Y[:,count] = np.reshape(im_diff_y,[N])
			count += 1
	return D_X, D_Y		

#  Test Unit 
# x_im = np.ones([32,32],dtype=np.complex128)
# x_im[8:25,8:24] = 1.5 - 0.5j
# x = np.reshape(x_im,[1024,1])
# A = np.random.randn(300,1024) + 1j*np.random.randn(300,1024)
# y = np.matmul(A,x)

# x_recon = ADMM_TV_complex(A,y,0.1,x)
# plt.subplot(2,2,1)
# plt.imshow(np.reshape(np.real(x),[32,32]))
# plt.subplot(2,2,2)
# plt.imshow(np.reshape(np.real(x_recon),[32,32]))
# plt.subplot(2,2,3)
# plt.imshow(np.reshape(np.imag(x),[32,32]))
# plt.subplot(2,2,4)
# plt.imshow(np.reshape(np.imag(x_recon),[32,32]))
# plt.show()

