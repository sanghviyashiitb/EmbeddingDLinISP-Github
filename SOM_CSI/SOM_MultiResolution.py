
import numpy as np
from util_functions import SOM_Stage_I, convert_w_to_x, convert_x_to_w
import numpy.linalg as LA
import torch
from multiprocessing import Pool
from functools import partial
import setup_functions
import util_functions
from util_cgfft import G_D_into_x, construct_g_D
from ContrastSourceNet import ContrastSourceNet_16_MultiScale_2, ContrastSourceNet_24_MultiScale_2
from SOM import SOM_SingularValues_Morozov, SOM_Stage_II_CGFFT_TV, TSOM_withCGFFT_TV
import cv2
import util_cgfft
MODEL_L16_FILE = './best_models_yet/ContrastSourceNet_noisydata_25SNR_L16.pth'

def TSOM_MultiResolution_without_DL(y, L1, L2, R1, R2, d, k, SNR, M_0, lambda_D):

	M, V = np.shape(y)[0], np.shape(y)[1]
	n1, n2 = d/L1, d/L2 
	pos_D_L1, pos_D_L2 = setup_functions.gen_pos_D(d,L1,n1), setup_functions.gen_pos_D(d,L2,n2)
	pos_S, pos_Tx = setup_functions.gen_pos_S(R1, M, d), setup_functions.gen_pos_Tx(R2, V, d)

	e_L1, e_L2 = setup_functions.gen_e(k, pos_D_L1, pos_Tx), setup_functions.gen_e(k, pos_D_L2, pos_Tx)
	G_S_L1, G_S_L2 = util_functions.construct_G_S(pos_D_L1, pos_S, k, n1), util_functions.construct_G_S(pos_D_L2, pos_S, k, n2)

	if L1 == 16:
		cs_net = ContrastSourceNet_16_MultiScale_2(V)
		cs_net.load_state_dict(torch.load(MODEL_L16_FILE))
	if L1 == 24:
		cs_net = ContrastSourceNet_24_MultiScale_2(V)
		cs_net.load_state_dict(torch.load(MODEL_L24_FILE))

	# SVD of G_S matrix
	U_L1, s_L1, V_L1h = np.linalg.svd(G_S_L1, full_matrices=False)
	S_L1 = np.diag(s_L1)
	V_L1 = V_L1h.conj().T

	U_L2, s_L2, V_L2h = np.linalg.svd(G_S_L2, full_matrices=False)
	S_L2 = np.diag(s_L2)
	V_L2 = V_L2h.conj().T

	# Guess obtained from L1
	# Obtain rowspace components of Contrast source from the noisy measurements, push it through network, L = 16
	w_RS_L1, _ = SOM_SingularValues_Morozov(y, G_S_L1, U_L1, S_L1, V_L1, 25) 
	
	g_D, g_D_fft, g_D_fft_conj = construct_g_D(pos_D_L1, k, n1)
	G_D_L1 = np.load('G_D/G_D_%d.npy'%(L1))
	# Convert w_DL to corresponding contrast x_DL
	d_0 =  e_L1 + G_D_into_x(g_D_fft, w_RS_L1)
	x_DL = np.reshape( np.sum(d_0.conj()*w_RS_L1, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (L1*L1,1) )
	# Use SOM at low-resolution for initial guess
	x_DL_SOM, _, _ = TSOM_withCGFFT_TV(y, G_S_L1, G_D_L1, g_D_fft, g_D_fft_conj, e_L1, w_RS_L1, x_DL, False, 256, 2000, 25, 0)
	x_im_L = np.reshape(x_DL_SOM,[L1,L1])
	x_DL_L2_real = cv2.resize(np.real(x_im_L),dsize=(L2,L2),interpolation=cv2.INTER_CUBIC)
	x_DL_L2_imag = cv2.resize(np.imag(x_im_L),dsize=(L2,L2),interpolation=cv2.INTER_CUBIC)
	
	# Use guess from low resolution, interpolate and use as initial guess in high resolution SOM 
	x_DL_L2 = np.reshape(x_DL_L2_real + 1j*x_DL_L2_imag,[L2*L2,1])

	g_D, g_D_fft, g_D_fft_conj = construct_g_D(pos_D_L2, k, n2)
	G_D_L2 = np.load('G_D/G_D_%d.npy'%(L2))
	# Convert w_DL to corresponding contrast x_DL
	# w_DL_L2, _	= util_functions.convert_x_to_w(x_DL_L2, G_S_L2, G_D_L2, e_L2)
	_, w_DL_L2 = util_cgfft.cg_fft_forward_problem(x_DL_L2, G_S_L2, g_D_fft, e_L2, 1e-6, e_L2, 1000)
	x_L2, _, cost_list_DL = TSOM_withCGFFT_TV(y, G_S_L2, G_D_L2, g_D_fft, g_D_fft_conj, e_L2, w_DL_L2, x_DL_L2, True, M_0, 2000, 25, lambda_D)
	x_L2 = np.real(x_L2)*(np.real(x_L2) >= 0) + 1j*np.imag(x_L2)*(np.imag(x_L2) <= 0)

	return x_L2, cost_list_DL

def TSOM_MultiResolution_with_DL(y, L1, L2, R1, R2, d, k, SNR, M_0, lambda_D):

	M, V = np.shape(y)[0], np.shape(y)[1]
	n1, n2 = d/L1, d/L2 
	pos_D_L1, pos_D_L2 = setup_functions.gen_pos_D(d,L1,n1), setup_functions.gen_pos_D(d,L2,n2)
	pos_S, pos_Tx = setup_functions.gen_pos_S(R1, M, d), setup_functions.gen_pos_Tx(R2, V, d)

	e_L1, e_L2 = setup_functions.gen_e(k, pos_D_L1, pos_Tx), setup_functions.gen_e(k, pos_D_L2, pos_Tx)
	G_S_L1, G_S_L2 = util_functions.construct_G_S(pos_D_L1, pos_S, k, n1), util_functions.construct_G_S(pos_D_L2, pos_S, k, n2)

	cs_net = ContrastSourceNet_16_MultiScale_2(V)
	cs_net.load_state_dict(torch.load(MODEL_L16_FILE))

	# SVD of G_S matrix
	U_L1, s_L1, V_L1h = np.linalg.svd(G_S_L1, full_matrices=False)
	S_L1 = np.diag(s_L1)
	V_L1 = V_L1h.conj().T

	U_L2, s_L2, V_L2h = np.linalg.svd(G_S_L2, full_matrices=False)
	S_L2 = np.diag(s_L2)
	V_L2 = V_L2h.conj().T

	# Guess obtained from L1
	# Obtain rowspace components of Contrast source from the noisy measurements, push it through network, L = 16
	w_RS_L1, _ = SOM_SingularValues_Morozov(y, G_S_L1, U_L1, S_L1, V_L1, SNR) 
	CSImage_input_L1 = np.expand_dims( util_functions.convert_w_to_CSImage(w_RS_L1),axis = 0) 
	CSImage_output_L1 = cs_net( torch.Tensor(CSImage_input_L1) ).detach().numpy()
	w_DL = util_functions.convert_CSImage_to_w(np.squeeze(CSImage_output_L1, axis=0))
	
	g_D, g_D_fft, g_D_fft_conj = construct_g_D(pos_D_L1, k, n1)
	G_D_L1 = np.load('G_D/G_D_%d.npy'%(L1))
	# Convert w_DL to corresponding contrast x_DL
	d_0 =  e_L1 + G_D_into_x(g_D_fft, w_DL)
	x_DL = np.reshape( np.sum(d_0.conj()*w_DL, axis = 1)/np.sum(d_0.conj()*d_0,axis=1), (L1*L1,1) )
	# Use SOM at low-resolution for initial guess
	x_DL_SOM, _, _ = TSOM_withCGFFT_TV(y, G_S_L1, G_D_L1, g_D_fft, g_D_fft_conj, e_L1, w_DL, x_DL, False, 256, 2000, SNR, 0)
	x_im_L = np.reshape(x_DL_SOM,[L1,L1])
	x_DL_L2_real = cv2.resize(np.real(x_im_L),dsize=(L2,L2),interpolation=cv2.INTER_CUBIC)
	x_DL_L2_imag = cv2.resize(np.imag(x_im_L),dsize=(L2,L2),interpolation=cv2.INTER_CUBIC)
	
	# Use guess from low resolution, interpolate and use as initial guess in high resolution SOM 
	x_DL_L2 = np.reshape(x_DL_L2_real + 1j*x_DL_L2_imag,[L2*L2,1])

	g_D, g_D_fft, g_D_fft_conj = construct_g_D(pos_D_L2, k, n2)
	G_D_L2 = np.load('G_D/G_D_%d.npy'%(L2))
	# Convert w_DL to corresponding contrast x_DL
	_, w_DL_L2 = util_cgfft.cg_fft_forward_problem(x_DL_L2, G_S_L2, g_D_fft, e_L2, 1e-6, e_L2, 1000)
	x_L2, _, cost_list_DL = TSOM_withCGFFT_TV(y, G_S_L2, G_D_L2, g_D_fft, g_D_fft_conj, e_L2, w_DL_L2, x_DL_L2, True, M_0, 2000, SNR, lambda_D)
	x_L2 = np.real(x_L2)*(np.real(x_L2) >= 0) + 1j*np.imag(x_L2)*(np.imag(x_L2) <= 0)

	return x_L2, cost_list_DL
