import numpy as np
from util_functions import SOM_Stage_I, convert_w_to_x, convert_x_to_w, convert_w_to_x_PenaltyPositive
import numpy.linalg as LA
from multiprocessing import Pool
from functools import partial
from scipy import fftpack
import scipy.linalg as la
from util_cgfft import G_D_into_x
from ADMM_TV import ADMM_TV_complex,get_TV_mat
import time
# Imported for ADMM, dealing with very sparse matrices here
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

def SOM_SingularValues_Morozov(s, G_S, U, S, V, SNR):
	M = np.shape(s)[0]
	noise_power = 10**(-SNR/10)*(np.linalg.norm(s,'fro')**2)

	residue = np.inf
	for sing_values in range(1,M+1):
		w_rs = SOM_Stage_I(U,S,V,s,sing_values)
		residue = np.linalg.norm(s - np.matmul(G_S, w_rs),'fro')**2
		if residue < noise_power:
			break

	print('Number of Singular Values used: %d'%(sing_values))
	return w_rs, sing_values

def SOM_Stage_II_CGFFT_TV(G_S, g_D_fft, g_D_fft_conj, s, e, w_source, V_N, x_init, alpha_init, max_iter, lambda_D):
	V = np.shape(s)[1]
	M = np.shape(s)[0]
	N = np.shape(G_S)[1]
	K = np.shape(V_N)[1]
	G_S_V_N = np.matmul(G_S, V_N)
	cost_list = []
	MAX_ADMM_ITER = 20
	# Difference operator
	L = np.int32((N)**0.5)
	D_X, D_Y = get_TV_mat(L)
	D_R = np.concatenate((D_X, D_Y, np.zeros([2*N,N])))
	D_I = np.concatenate((np.zeros([2*N,N]),D_X, D_Y))
	D = np.concatenate((D_R,D_I),axis=1)
	D_T_D = csc_matrix(np.matmul(D.T, D))
	rho = 0.01 				# Used in ADMM stuff
	kappa = lambda_D/rho 	# Used in ADMM stuff
	# Normalization constants (used in cost function)
	eta_field = np.reshape(np.sum(np.abs(s)**2, axis = 0)**0.5,(1,V))
	eta_curr = np.reshape(np.sum(np.abs(w_source)**2, axis = 0)**0.5,(1,V))

	# Initialization of residuals
	cost = np.inf
	x = x_init
	alpha = alpha_init
	X = np.diag(x[:,0])
	w_nullspace = np.matmul(V_N, alpha)
	A_alpha = np.matmul(X, G_D_into_x(g_D_fft, w_nullspace)) - w_nullspace
	b = w_source - np.matmul(X, e) - np.matmul( X,  G_D_into_x(g_D_fft, w_source) )
	field_residue = np.matmul(G_S, w_source) + np.matmul( G_S_V_N , alpha) - s
	curr_residue = 	A_alpha - b	
	cost_init = LA.norm(field_residue/eta_field,'fro')**2 + LA.norm(curr_residue/eta_curr,'fro')**2
	grad_alpha_field = np.matmul( G_S_V_N.conj().T , field_residue)/eta_field**2
	grad_alpha_curr  = np.matmul( V_N.conj().T , G_D_into_x(g_D_fft_conj, np.matmul(X.conj().T, curr_residue) ) - curr_residue )/eta_curr**2 
	grad_alpha = grad_alpha_field + grad_alpha_curr
	
	del_alpha = np.zeros((K,V),dtype=np.complex128)
	del_alpha_new = np.zeros((K,V),dtype=np.complex128)
	step_alpha = np.ones((1,V),dtype=np.float64)

	# Beginning of SOM second stage optimization procedure
	start_time_SOM = time.time()
	for iteration in range(max_iter):
		start_time_SOM_iter = time.time()
		# Evaluate gradient for alpha of each view
		X = np.diag(x[:,0])
		w_nullspace = np.matmul(V_N, alpha)
		A_alpha = np.matmul(X, G_D_into_x(g_D_fft, w_nullspace)) - w_nullspace
		b = w_source - np.matmul(X, e) - np.matmul( X,  G_D_into_x(g_D_fft, w_source) )
		field_residue = np.matmul(G_S, w_source) + np.matmul( G_S_V_N , alpha) - s
		curr_residue = 	A_alpha - b
		# Calculation of gradient of cost function
		grad_alpha_prev = grad_alpha
		grad_alpha_field = np.matmul( G_S_V_N.conj().T , field_residue)/eta_field**2
		grad_alpha_curr  = np.matmul( V_N.conj().T , G_D_into_x(g_D_fft_conj, np.matmul(X.conj().T, curr_residue) ) - curr_residue )/eta_curr**2 
		grad_alpha = grad_alpha_field + grad_alpha_curr
	
		# Evalute descent direction and step size using Polack Ribiere CG Rule
		for v in range(V):
			k_v = np.real( np.sum( (grad_alpha[:,v] - grad_alpha_prev[:,v] ).conj()*grad_alpha[:,v]) )/(np.sum(np.abs(grad_alpha_prev[:,v])**2))
			del_alpha_v = np.reshape(grad_alpha[:,v] + k_v*del_alpha[:,v],(K,1))
			# Evaluation of step size
			field_residue_v = np.reshape(field_residue[:,v],(M,1))
			curr_residue_v = np.reshape(curr_residue[:,v],(N,1))
			del_w_null_v = np.matmul(V_N, del_alpha_v)
			A_del_w_null_v = np.matmul(X, G_D_into_x(g_D_fft, del_w_null_v)) - del_w_null_v
			A_v = LA.norm(np.matmul(G_S , del_w_null_v))**2/(eta_field[0,v]**2) + LA.norm( A_del_w_null_v )**2/(eta_curr[0,v]**2)
			B_v = np.real(np.sum( np.matmul( G_S , del_w_null_v).conj()*field_residue_v))/(eta_field[0,v]**2)
			B_v += np.real(np.sum(A_del_w_null_v.conj()*curr_residue_v))/(eta_curr[0,v]**2)
			
			del_alpha_new[:,v] = np.reshape(del_alpha_v, (K))
			step_alpha[:,v] = -B_v/A_v
		del_alpha = del_alpha_new
		# Update alpha
		alpha += step_alpha*del_alpha
		# Update contrast source
		w = w_source + np.matmul(V_N, alpha)
		# Update internal fields
		d = e + G_D_into_x(g_D_fft, w)

		if lambda_D == 0:
			x_num = d.conj()*w/(eta_curr**2)
			x_den = d.conj()*d/(eta_curr**2)
			x = np.reshape( np.sum(x_num, axis = 1)/np.sum(x_den, axis = 1), [N,1])
		else:
			# # # # # 
			# update contrast using ADMM TV
			# Note that in this case A is a diagonal, hence only represented as a vector to avoid memory issues 
			# Will be an entire sub-block, splits the problem into real and imaginary
			cost_ADMM = np.inf
			x = np.concatenate((np.real(x),np.imag(x)))
			z = np.matmul(D, x)
			u = np.zeros([4*N,1])
			temp_d = np.sum(abs(d/eta_curr)**2,axis=1)
			A = diags(np.concatenate((temp_d,temp_d)),0,format="csc")
			x_pinv_A = csc_matrix(A + rho*D_T_D)
			d_conj_w = np.reshape(np.sum((d.conj()*w)/eta_curr**2	,axis=1),[N,1])
			b_temp_1 = np.concatenate((np.real(d_conj_w),np.imag(d_conj_w)))
			# Solving scaled version of ADMM
			start_time_ADMM = time.time()
			for iteration_ADMM in range(MAX_ADMM_ITER):
				# X update
				x = np.reshape(spsolve(x_pinv_A, b_temp_1 + rho*np.matmul(D.T, z-u) ),[2*N,1])
				# Z update
				z_temp = np.matmul(D,x) + u
				z = (z_temp >= kappa)*(z_temp - kappa) + (z_temp <= -kappa)*(z_temp+kappa)
				# U update
				u = u + (np.matmul(D, x) - z)

				x_complex = x[:N,:] + 1j*x[N:,:]
				residue, regularizer = np.linalg.norm((d*np.repeat(x_complex,V,axis=1)-w)/eta_curr,'fro'), np.linalg.norm(np.matmul(D,x),1)
				prev_cost_ADMM = cost_ADMM
				cost_ADMM = residue**2 + lambda_D*regularizer		
				if 0.99*prev_cost_ADMM < cost_ADMM:
					break		
			end_time_ADMM = time.time()
			print("ADMM_Elapsed Time: %.3f"%(end_time_ADMM-start_time_ADMM))
			x_real = x
			x = x[:N,:] + 1j*x[N:,:]
			# End of contrast update using ADMM TV
			# # # # #		
		X = np.diag(x[:,0])
		w_nullspace = np.matmul(V_N, alpha)
		A_alpha = np.matmul(X, G_D_into_x(g_D_fft, w_nullspace)) - w_nullspace
		b = w_source - np.matmul(X, e) - np.matmul( X,  G_D_into_x(g_D_fft, w_source) )
		field_residue = np.matmul(G_S, w_source) + np.matmul( G_S_V_N , alpha) - s
		curr_residue = 	A_alpha - b
		prev_cost = cost
		cost = LA.norm(field_residue/eta_field,'fro')**2 + LA.norm(curr_residue/eta_curr,'fro')**2 #+ lambda_D*LA.norm(np.matmul(D,x_real),1)
		cost_list.append(cost)
		print('Iteration: %d, Cost function: %0.5f'%(iteration, cost))
		if cost < 1e-4 or abs(cost - prev_cost) < 1e-4*prev_cost:
			break
		
		end_time_SOM_iter = time.time()
		print('SOM Iteration time: %0.3f'%(end_time_SOM_iter-start_time_SOM_iter))
	end_time_SOM = time.time()
	print('Time elapsed: ',end_time_SOM-start_time_SOM)
	return x, w, cost_list

def TSOM_withCGFFT_TV(s, G_S, G_D, g_D_fft, g_D_fft_conj, e, w_0, x_0, use_x_0, sing_values_D, max_iter, SNR, lambda_D):
	M, V = np.shape(s)[0], np.shape(s)[1]
	N = np.shape(G_S)[1]

	# SVD of G_S matrix
	U_S, s_S, V_Sh = np.linalg.svd(G_S, full_matrices=True)
	S_S = np.diag(s_S)
	V_S = V_Sh.conj().T
	# Choosing number of singular values to be used and calculating signal (or source) subspace component of TSOM
	w_source, sing_values_S = SOM_SingularValues_Morozov(s, G_S, U_S, S_S, V_S, SNR)
	V_S_signal, V_S_ambi = V_S[:,:sing_values_S], V_S[:,sing_values_S:N]

	# SVD of G_D matrix			
	U_D, s_D, V_Dh = np.linalg.svd(G_D, full_matrices=True)
	S_D = np.diag(s_D)
	V_D = V_Dh.conj().T
	V_D_ambi = V_D[:,:sing_values_D]
	
	V_D = np.matmul(np.matmul(V_S_ambi, V_S_ambi.conj().T), V_D_ambi)
	alpha_init = np.matmul(np.linalg.pinv(V_D), w_0)
	# As described in Original SOM paper
	if use_x_0:
		x_init = x_0
	else:
		d_0 =  e + G_D_into_x(g_D_fft, w_0)
		x_init = np.reshape(np.sum(d_0.conj()*w_0, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (N,1) )

	x, w, cost_list = SOM_Stage_II_CGFFT_TV(G_S, g_D_fft, g_D_fft_conj, s, e, w_source, V_D, x_init, alpha_init, max_iter, lambda_D)
	return x, w, cost_list


def TSOM_withCGFFT_TV_HF(s, G_S, G_D, g_D_fft, g_D_fft_conj, e, w_0, x_0, use_x_0, sing_values_D, max_iter, SNR, lambda_D):
	M, V = np.shape(s)[0], np.shape(s)[1]
	N = np.shape(G_S)[1]
	L = np.int32(N**0.5)
	# SVD of G_S matrix
	U_S, s_S, V_Sh = np.linalg.svd(G_S, full_matrices=True)
	S_S = np.diag(s_S)
	V_S = V_Sh.conj().T
	# Choosing number of singular values to be used and calculating signal (or source) subspace component of TSOM
	w_source, sing_values_S = SOM_SingularValues_Morozov(s, G_S, U_S, S_S, V_S, SNR)
	V_S_signal, V_S_ambi = V_S[:,:sing_values_S], V_S[:,sing_values_S:N]

	# # SVD of G_D matrix			
	# U_D, s_D, V_Dh = np.linalg.svd(G_D, full_matrices=True)
	# S_D = np.diag(s_D)
	# V_D = V_Dh.conj().T
	V_D = np.load('G_D/G_D_Vd_%d_600MHz.npy'%(L))

	V_D_ambi = V_D[:,:sing_values_D]
	
	V_D = np.matmul(np.matmul(V_S_ambi, V_S_ambi.conj().T), V_D_ambi)
	alpha_init = np.matmul(np.linalg.pinv(V_D), w_0)
	# As described in Original SOM paper
	if use_x_0:
		x_init = x_0
	else:
		d_0 =  e + G_D_into_x(g_D_fft, w_0)
		x_init = np.reshape(np.sum(d_0.conj()*w_0, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (N,1) )

	x, w, cost_list = SOM_Stage_II_CGFFT_TV(G_S, g_D_fft, g_D_fft_conj, s, e, w_source, V_D, x_init, alpha_init, max_iter, lambda_D)
	return x, w, cost_list

def TSOM_withCGFFT_TV_LF(s, G_S, G_D, g_D_fft, g_D_fft_conj, e, w_0, x_0, use_x_0, sing_values_D, max_iter, SNR, lambda_D):
	M, V = np.shape(s)[0], np.shape(s)[1]
	N = np.shape(G_S)[1]
	L = np.int32(N**0.5)
	# SVD of G_S matrix
	U_S, s_S, V_Sh = np.linalg.svd(G_S, full_matrices=True)
	S_S = np.diag(s_S)
	V_S = V_Sh.conj().T
	# Choosing number of singular values to be used and calculating signal (or source) subspace component of TSOM
	w_source, sing_values_S = SOM_SingularValues_Morozov(s, G_S, U_S, S_S, V_S, SNR)
	V_S_signal, V_S_ambi = V_S[:,:sing_values_S], V_S[:,sing_values_S:N]

	# # SVD of G_D matrix			
	# U_D, s_D, V_Dh = np.linalg.svd(G_D, full_matrices=True)
	# S_D = np.diag(s_D)
	# V_D = V_Dh.conj().T
	V_D = np.load('G_D/G_D_Vd_%d_300MHz.npy'%(L))

	V_D_ambi = V_D[:,:sing_values_D]
	
	V_D = np.matmul(np.matmul(V_S_ambi, V_S_ambi.conj().T), V_D_ambi)
	alpha_init = np.matmul(np.linalg.pinv(V_D), w_0)
	# As described in Original SOM paper
	if use_x_0:
		x_init = x_0
	else:
		d_0 =  e + G_D_into_x(g_D_fft, w_0)
		x_init = np.reshape(np.sum(d_0.conj()*w_0, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (N,1) )

	x, w, cost_list = SOM_Stage_II_CGFFT_TV(G_S, g_D_fft, g_D_fft_conj, s, e, w_source, V_D, x_init, alpha_init, max_iter, lambda_D)
	return x, w, cost_list

def TSOM_withCGFFT_TV_100(s, G_S, G_D, g_D_fft, g_D_fft_conj, e, w_0, x_0, use_x_0, sing_values_D, max_iter, SNR, lambda_D):
	M, V = np.shape(s)[0], np.shape(s)[1]
	N = np.shape(G_S)[1]
	L = np.int32(N**0.5)
	# SVD of G_S matrix
	U_S, s_S, V_Sh = np.linalg.svd(G_S, full_matrices=True)
	S_S = np.diag(s_S)
	V_S = V_Sh.conj().T
	# Choosing number of singular values to be used and calculating signal (or source) subspace component of TSOM
	w_source, sing_values_S = SOM_SingularValues_Morozov(s, G_S, U_S, S_S, V_S, SNR)
	V_S_signal, V_S_ambi = V_S[:,:sing_values_S], V_S[:,sing_values_S:N]

	# # SVD of G_D matrix			
	# U_D, s_D, V_Dh = np.linalg.svd(G_D, full_matrices=True)
	# S_D = np.diag(s_D)
	# V_D = V_Dh.conj().T
	# try:
		# V_D = np.load('G_D/G_D_Vd_%d_100MHz.npy'%(L))
	# except:
	U_D, s_D, V_Dh = np.linalg.svd(G_D, full_matrices=True)
	S_D = np.diag(s_D)
	V_D = V_Dh.conj().T
	np.save('G_D/G_D_Vd_%d_100MHz.npy'%(L),V_D)

	V_D_ambi = V_D[:,:sing_values_D]
	
	V_D = np.matmul(np.matmul(V_S_ambi, V_S_ambi.conj().T), V_D_ambi)
	alpha_init = np.matmul(np.linalg.pinv(V_D), w_0)
	# As described in Original SOM paper
	if use_x_0:
		x_init = x_0
	else:
		d_0 =  e + G_D_into_x(g_D_fft, w_0)
		x_init = np.reshape(np.sum(d_0.conj()*w_0, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (N,1) )

	x, w, cost_list = SOM_Stage_II_CGFFT_TV(G_S, g_D_fft, g_D_fft_conj, s, e, w_source, V_D, x_init, alpha_init, max_iter, lambda_D)
	return x, w, cost_list