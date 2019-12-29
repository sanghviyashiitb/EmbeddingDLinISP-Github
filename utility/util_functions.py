import numpy as np
import cv2
import scipy.special as sc
import numpy.linalg as linalg
from scipy.interpolate import interp1d
def forward_solver(x, G_S, G_D, e):
	N = np.shape(G_D)[0] # Number of grids
	M = np.shape(G_S)[0] # Number of receivers
	V = np.shape(e)[1]   # Number of views
	# Scattered field given by the relation
	# s_v = G_S*X*L_X*e_v
	# where L_X = (I - G_D*X)^(-1)
	s = np.empty([M, V],dtype=np.complex128)
	X = np.diag(x[:,0])
	L_X = linalg.inv(np.eye(N)- np.matmul(G_D,X))
	d = np.matmul(L_X,e)		
	for v in range(V):
		w_v = x*np.reshape(d[:,v],[N,1])
		s[:,v] = np.reshape(np.matmul(G_S,w_v),M)
	return s


def construct_G_D(pos_D, k, n):
	# pos_D : N X 2 vector containing position of each grid
	# n : size of each grid
	# k : wavevector
	a = (n**2/np.pi)**0.5
	N = np.shape(pos_D)[0]

	G_D = np.zeros([N,N],dtype=np.complex128)
	for i in range(N):
		for j in range(N):
			if i == j:
				G_D[i,j] = -1j*0.5*(np.pi*k*a*sc.hankel2(1,k*a) - 2*1j)
			else:
				rho_ij = linalg.norm(pos_D[i,:]-pos_D[j,:],2)
				G_D[i,j] = -1j*0.5*np.pi*k*a*sc.j1(k*a)*sc.hankel2(0,k*rho_ij)
	return G_D

def construct_G_S(pos_D, pos_S, k, n):
	M = np.shape(pos_S)[0]
	N = np.shape(pos_D)[0]
	G_S = np.zeros([M,N],dtype =np.complex128)	
	a = (n**2/np.pi)**0.5

	for i in range(M):
		for j in range(N):
			rho_ij = linalg.norm(pos_S[i,:]-pos_D[j,:],2)
			G_S[i,j] = -1j*0.5*np.pi*k*a*sc.j1(k*a)*sc.hankel2(0,k*rho_ij)
	return G_S

def field_error(s_1, s_2):
	M, V = np.shape(s_1)[0], np.shape(s_1)[1]
	err = 0
	for v in range(V):
		err += (linalg.norm(s_1[:,v]-s_2[:,v])**2)/(linalg.norm(s_1[:,v])**2)
	err /= V

	return err

# BornIterativeMethod 
def BornIterativeMethod(s, G_S, G_D, e, lambda_2, x_0, regularization = 'L2', max_iter=20):
	M, N, V = np.shape(s)[0], np.shape(G_D)[0], np.shape(s)[1]
	A = np.empty([M*V,N],dtype=np.complex)
	y = np.empty([M*V,1],dtype=np.complex)
	for v in range(V):
		y[v*M:(v+1)*M,0] = s[:,v]
	x = x_0
	for iter in range(max_iter):

		L_X = linalg.inv(np.eye(N)- np.matmul(G_D,np.diag(x[:,0])))
		d = np.matmul(L_X,e)
		print(np.shape(d))
		for v in range(V):
			A[v*M:(v+1)*M,:] = np.matmul(G_S,np.diag(d[:,v]))

		print('Iteration Number: %d'%(iter))
		err = np.linalg.norm(np.matmul(A,x) - y)/np.linalg.norm(y)
		print('Forward Error: %0.3f'%(err))

		A_H = A.conj().T

		x = np.matmul(linalg.inv(np.matmul(A_H,A) + lambda_2*np.eye(N)),np.matmul(A_H,y))

		if np.linalg.norm(x[:,0]-x_prev[:,0],np.inf) < 1e-2:
			print('Convergence achieved. Breaking after %d iterations.'%(iter))
			break
	return x

def convert_w_to_CSImage(w):
	# Splits the complex image channels into real and complex image channels
	L, V = np.int32(np.shape(w)[0]**0.5), np.shape(w)[1]
	CSImage = np.empty((2*V,L,L),dtype=np.float32)
	for v in range(V):
		im_v = np.reshape(w[:,v], (L,L))
		CSImage[v*2,:,:], CSImage[v*2 + 1,:,:] = np.real(im_v), np.imag(im_v)
	return CSImage

def convert_CSImage_to_w(CSImage):
	L, V = np.shape(CSImage)[1], np.int32(np.shape(CSImage)[0]/2)
	w = np.empty((L*L,V),dtype=np.complex128)
	for v in range(V):
		w_v_R, w_v_I = np.reshape(CSImage[2*v,:,:], L*L), np.reshape(CSImage[2*v+1,:,:], L*L)
		w[:,v] = w_v_R + 1j*w_v_I
	return w

def convert_x_to_w(x, G_S, G_D, e):
	N = np.shape(G_D)[0] # Number of grids
	M = np.shape(G_S)[0] # Number of receivers
	V = np.shape(e)[1]   # Number of views
	# Scattered field given by the relation
	# s_v = G_S*X*L_X*e_v
	# where L_X = (I - G_D*X)^(-1)
	s = np.empty([M, V],dtype=np.complex128)
	w = np.empty([N, V],dtype=np.complex128)

	X = np.diag(x[:,0])
	L_X = linalg.inv(np.eye(N)- np.matmul(G_D,X))
	d = np.matmul(L_X,e)
	
	# w = d*np.repeat(x,V,axis=1)
	for v in range(V):
		w_v = x*np.reshape(d[:,v],[N,1])
		w[:,v] = np.reshape(w_v,(N))
		s[:,v] = np.reshape(np.matmul(G_S,w_v),M)
	return w, s

def SOM_Stage_I(U1,S1,V1,s,sing_values):
	# sing_values: How many components of the row space do you want to use?
	# U_m,S_m,V_m = svd(G_S)
	# b := The row space coefficients for each view (dim: sing_values X V)
	U_r = U1[:,:sing_values] 
	S_r = S1[:sing_values,:sing_values]
	V_r = V1[:,:sing_values]
	b = np.matmul(np.linalg.inv(S_r), np.matmul(U_r.conj().T,s))
	w_rowspace = np.matmul(V_r, b)
	return w_rowspace

def convert_w_to_x(w, G_S, G_D, e):
	N, V = np.shape(G_D)[0], np.shape(e)[1]
	d = e + np.matmul(G_D,w) 
	# equivalent to solving Ax = y overdetermined problem
	# A and y are to be defined in the next few lines
	A = np.zeros( (N,N), dtype=np.complex128)
	y = np.zeros( (N,1), dtype=np.complex128)
	for v in range(V):
		D_v = np.diag(d[:,v])
		A += np.matmul(D_v.conj().T, D_v)
		y += np.matmul(D_v.conj().T, np.reshape(w[:,v],[N,1]))
	x = np.matmul( np.linalg.pinv(A), y )
	return x

def convert_w_to_x_PenaltyPositive(w, G_S, G_D, e, lambda_penalty):
	N, V = np.shape(G_D)[0], np.shape(e)[1]
	d = e + np.matmul(G_D,w) 
	
	d_abs = np.reshape(np.sum(np.abs(d)**2, axis=1),[N,1])
	c = np.reshape(np.sum(d.conj()*w, axis=1),[N,1])
	
	c_real = np.real(c)
	x_real = (c_real <= 0)*(c_real)/(d_abs + lambda_penalty) + (c_real >= 0)*(c_real)/d_abs

	c_imag = np.imag(c)
	x_imag = (c_imag >= 0)*(c_imag)/(d_abs + lambda_penalty) + (c_imag <= 0)*(c_imag)/d_abs	

	x = x_real + 1j*x_imag
	return x

def convert_batch_to_CSImageandY(batch, L, G_S, G_D, e, max_contrast, min_contrast = 0.0, randomize=True):
	BATCH_SIZE = np.shape(batch)[0]
	M, V = np.shape(G_S)[0], np.shape(e)[1]
	CS_Image = np.empty([BATCH_SIZE,2*V,L,L])
	Y = np.empty([BATCH_SIZE,M,V],dtype=np.complex128)
	# Function prepares the batch from MNIST for input to ContrastSourceNet
	for idx in range(BATCH_SIZE):
		# Extract a single contrast and resize it to the specified dimensions
		im = np.squeeze(batch[idx,:,:,:],axis=0)
		im_resize = cv2.resize(np.real(im),(L,L)) + 1j*cv2.resize(np.imag(im),(L,L))
		x =  np.reshape(im_resize,(L*L,1))
		if randomize:
			contrast = np.around((max_contrast-min_contrast)*np.random.rand() + min_contrast, decimals=0)
		else:
			contrast = max_contrast
		x = contrast*x
		# Obtain the contrast source and scattered field 
		w, y = convert_x_to_w(x, G_S, G_D, e)
		
		# Convert contrast source to the network's format
		im_cs = convert_w_to_CSImage(w)
		CS_Image[idx,:,:,:] = im_cs
		Y[idx,:,:] = y

	return CS_Image, Y

def cubicinterp_contrastsource(CSImage, out_size):
	N, L_X, L_Y = np.shape(CSImage)[0], np.shape(CSImage)[1], np.shape(CSImage)[2]
	CSImage_out = np.empty((N,out_size[0],out_size[1]),dtype=np.float32)
	for idx in range(N):
		CSImage_out[idx,:,:] = cv2.resize(CSImage[idx,:,:],dsize=out_size,interpolation=cv2.INTER_CUBIC)

	return CSImage_out

def add_noise(signal, SNR):
	signal_shape = np.shape(signal)
	signal_power = np.linalg.norm(signal,'fro')**2
	sigma = ((10**(-SNR/10))*signal_power/np.prod(signal_shape))**0.5
	noise = sigma*np.random.randn(*signal_shape)

	return signal + noise

def shape_error(contrast, contrast_true):
	N = np.shape(contrast)[0]
	diff = np.divide(np.abs(contrast - contrast_true),np.abs(contrast_true + 1))
	err_total = np.sum(diff)/N
	err_internal = np.sum(diff*(abs(contrast_true)> 1e-3))/np.sum(np.asarray(abs(contrast_true)>1e-3,dtype=np.float32))
	return [err_internal, err_total]


def convert_batch_to_CSImageandY_1(batch, L, G_S, G_D, e, max_contrast, min_contrast = 0.0, randomize=True):
	BATCH_SIZE = np.shape(batch)[0]
	M, V = np.shape(G_S)[0], np.shape(e)[1]
	X_Image = np.empty((BATCH_SIZE,L,L),dtype=np.complex128)
	CS_Image = np.empty([BATCH_SIZE,2*V,L,L])
	Y = np.empty([BATCH_SIZE,M,V],dtype=np.complex128)
	# Function prepares the batch from MNIST for input to ContrastSourceNet
	for idx in range(BATCH_SIZE):
		# Extract a single contrast and resize it to the specified dimensions
		im = np.squeeze(batch[idx,:,:,:],axis=0)
		im_resize = cv2.resize(np.real(im),(L,L)) + 1j*cv2.resize(np.imag(im),(L,L))
		x =  np.reshape(im_resize,(L*L,1))
		if randomize:
			contrast = np.around((max_contrast-min_contrast)*np.random.rand() + min_contrast, decimals=0)
		else:
			contrast = max_contrast
		x = contrast*x
		X_Image[idx,:,:] = np.reshape(x,[L,L])
		# Obtain the contrast source and scattered field 
		w, y = convert_x_to_w(x, G_S, G_D, e)
		
		# Convert contrast source to the network's format
		im_cs = convert_w_to_CSImage(w)
		CS_Image[idx,:,:,:] = im_cs
		Y[idx,:,:] = y

	return CS_Image, Y, X_Image

def col_norm(vector):
	return np.sum(np.abs(vector)**2,0)**0.5


def interpolate_views(input_tensor, input_V, target_V):

	input_views = np.linspace(0,1,input_V)
	output_views = np.linspace(0,1,target_V)

	f_input = interp1d(input_views, input_tensor, axis = 0)
	output_tensor = f_input(output_views)

	return output_tensor

def convert_w_to_CSImage_withoutsplit(w):
	L, V = np.int32(np.shape(w)[0]**0.5), np.shape(w)[1]
	CSImage = np.empty((V,L,L),dtype=np.complex128)
	for v in range(V):
		im_v = np.reshape(w[:,v], (L,L))
		CSImage[v,:,:] = im_v
	return CSImage

def CSImage_to_w_withoutsplit(CSImage):
	L, V = np.shape(CSImage)[1], np.shape(CSImage)[0]
	w = np.empty((L*L,V),dtype=np.complex128)
	for v in range(V):
		w_v = np.reshape(CSImage[v,:,:], L*L)
		w[:,v] = w_v
	return w

def split_CSImage(CSImage):
	L, V = np.shape(CSImage)[1], np.shape(CSImage)[0]
	CSImage_output = np.empty((2*V,L,L),dtype=np.float32)
	for v in range(V):
		im_v = CSImage[v,:,:]
		CSImage_output[2*v,:,:], CSImage_output[2*v + 1,:,:]  = np.real(im_v), np.imag(im_v)
	return CSImage_output

def combine_CSImage(CSImage):
	L, V = np.shape(CSImage)[1], np.int32(np.shape(CSImage)[0]/2)
	CSImage_output = np.empty((V,L,L),dtype=np.complex128)
	for v in range(V):
		im_v_real = CSImage[2*v,:,:]
		im_v_imag = CSImage[2*v+1,:,:]
		CSImage_output[v,:,:] = im_v_real + 1j*im_v_imag
	return CSImage_output