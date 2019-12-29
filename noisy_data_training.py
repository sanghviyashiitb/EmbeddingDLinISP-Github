# Script for training the ContrastSourceNet
# Author: Yash Sanghvi

# A FEW NOTES ABOUT THE CODE TO UNDERSTAND IT BETTER
# The network takes into input the contrast source row-space components and outputs (or tries to output) the true contrast source.
# As a result it has estimated the null-space components, and denoised the row-space components
# CONTRAST SOURCE VECTORS AND HOW IT IS FED INTO THE NETWORK
# PyTorch doesn't have any functionality for complex numbers, 
# so each contrast source vector for a view is reshaped into an image with a single channel and then split into 2 channels, one for real and other imaginary parts 
# So whenever the contrast source is fed into the network, you can see the following code
# CSImage = util_functions.convert_w_to_CSImage(w)
# This CSImage has dimensions: [2V X L X L] and the network takes the input into the form: [BATCH_SIZE X 2V X L X L]
# So if only a single contrast source is fed into the input you need to exapnd the first dimension as follows:
# CSImage_input = np.expand_dims(CSImage, axis=0)
# This line will change dimension from [2V X L X L] to [1 X 2V X L X L]
import sys
sys.path.insert(0, './utility')
sys.path.insert(0, './SOM_CSI')
import numpy as np
import setup_functions
import util_functions
import generate_shapes
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from ContrastSourceNet import ContrastSourceNet_16_MultiScale_2

# Setting up parameters for Inverse Scattering
wavelength = 0.75
k = 2*np.pi/wavelength # Wavevector
d = 2 # Size of imaging domain (in SI Units)
L = 16		 
n = d/L
R = 4
M = 32
V = 16
pos_D, pos_S, pos_Tx = setup_functions.gen_pos_D(d,L,n), setup_functions.gen_pos_S(R, M, d), setup_functions.gen_pos_Tx(R*1.5, V, d)
e = setup_functions.gen_e(k, pos_D, pos_Tx)
G_D, G_S = util_functions.construct_G_D(pos_D, k, n), util_functions.construct_G_S(pos_D, pos_S, k, n)

# Hyperparameters for network training
BATCH_SIZE = 40
BATCH_SIZE_TEST = 400
RESTART = True # Set to True if you want to retrain the network from scratch
MAX_EPOCH = 50
max_contrast = 7.0 
min_contrast = 1.0	
LEARN_RATE = 1e-4
SNR = 25
sing_values = 19 # Determined from Morozov's principle. Should be modified for a different SNR value

# Initializing a network 
cs_net =  ContrastSourceNet_16_MultiScale_2(V)
MODEL_L16_FILE = './best_models_yet/ContrastSourceNet_noisydata_25SNR_L16_8.pth'
if not RESTART:
	cs_net.load_state_dict(torch.load(MODEL_L16_FILE))

# Loading the test and train dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,0),(1.0, 1.0))])
trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=1)

# Obtain a batch from the test dataset
for i, data in enumerate(testloader):
	batch_test, _ = data
	batch_test_numpy = batch_test.numpy()
	break

# SVD of G_S matrix
U1, s1, V1h = np.linalg.svd(G_S, full_matrices=False)
S1 = np.diag(s1)
V1 = V1h.conj().T

# Training code
# Loss function used: Mean Squared Error between the true and predicted CS, with each true CS L2 norm in the denominator
# for normalization
loss = nn.MSELoss()
def loss_fn(pred_cs, true_cs):
	true_cs_flat = true_cs.view(-1,2*V*L*L)
	pred_cs_flat = pred_cs.view(-1,2*V*L*L)
	return loss(pred_cs_flat, true_cs_flat)

# Optimizer to be used: Stochastic Gradient Descent with Momentum
optimizer = optim.Adam(cs_net.parameters(),lr = LEARN_RATE)

# Defining placeholders for input of the network
CSImage_input = np.empty((BATCH_SIZE,2*V,L,L),dtype=np.float32)
loss_list = [] # Keep track of the training loss, used in deciding whether or not to decrease the learning rate
# Training begins here!
iteration = 0
loss_avg = np.inf

cs_net.train()
for epoch in range(MAX_EPOCH):
	if epoch == 40:
		LEARN_RATE *= 0.2
		optimizer = optim.Adam(cs_net.parameters(),lr = LEARN_RATE)
	for i, data in enumerate(trainloader):
		# Obtain a batch from the train dataset
		batch, _ = data
		batch_numpy = batch.numpy()
		# Calculate the true contrast source and measurement data from the batch
		CSImage_true, Y = util_functions.convert_batch_to_CSImageandY(batch_numpy, L, G_S, G_D, e, max_contrast, min_contrast, True)
		# Convert the true and input contrast source into real images for it to be made ready for the network
		# Nothing fancy
		for idx in range(BATCH_SIZE):
			Y[idx,:,:] = util_functions.add_noise(Y[idx,:,:], SNR)
			CSImage_input[idx,:,:,:]  = util_functions.convert_w_to_CSImage(util_functions.SOM_Stage_I(U1,S1,V1,Y[idx,:,:],sing_values))
		
		# Set all the gradients equal to zero
		optimizer.zero_grad()
		# Forward pass the batch to network
		CSImage_output = cs_net(torch.Tensor(CSImage_input))
		# Loss function calculation
		loss_value = loss_fn(CSImage_output,torch.Tensor(CSImage_true))

		# Calculate gradients for all network parameters
		loss_value.backward()
		# Perform gradient update steps
		optimizer.step()
		loss_list.append(loss_value)

		if np.mod(iteration,10) == 0:
			print('Iteration: %d, Loss: %.5f'%(iteration,loss_value))
						
		if np.mod(iteration,200) == 0 and iteration > 0:
			CSImage_true_test, Y_test = util_functions.convert_batch_to_CSImageandY(batch_test_numpy, L, G_S, G_D, e, max_contrast, min_contrast, True)
			# Obtaining row-space components from  SOM stage I
			CSImage_input_test = np.empty((BATCH_SIZE_TEST,2*V,L,L),dtype=np.float32)
			for idx in range(BATCH_SIZE_TEST):
				Y_test[idx,:,:] = util_functions.add_noise(Y_test[idx,:,:],SNR)
				CSImage_input_test[idx,:,:,:] = util_functions.convert_w_to_CSImage(util_functions.SOM_Stage_I(U1,S1,V1,Y_test[idx,:,:],sing_values))
			CSImage_output_test = cs_net(torch.Tensor(CSImage_input_test))
			loss_value_test = loss_fn(CSImage_output_test,torch.Tensor(CSImage_true_test))
			
			prev_loss_avg = loss_avg
			loss_avg = np.mean(np.asarray(loss_list,dtype=np.float32))
			loss_list = []
			print('---------------------------------------------------------------------------') 
			print('Loss (on Test batch): %.5f'%(loss_value_test))
			print('Averaged Loss: %.5f, Previous Average Loss: %.5f'%(loss_avg,prev_loss_avg))
			print('---------------------------------------------------------------------------') 
		iteration += 1
	torch.save(cs_net.state_dict(),'./best_models_yet/ContrastSourceNetTest_Epoch_%d.pth'%(epoch))
torch.save(cs_net.state_dict(),MODEL_L16_FILE)
