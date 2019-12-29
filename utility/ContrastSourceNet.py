import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastSourceNet_16(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_16, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 2*V, out_channels = 2*V*5, kernel_size = (3, 3), padding = 1)
		self.conv2 = nn.Conv2d(in_channels = 2*V*5, out_channels = 2*V*2, kernel_size = (5, 5), padding = 2)
		self.conv3 = nn.Conv2d(in_channels = 2*V*2, out_channels = 2*V, kernel_size = (3, 3), padding = 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.conv3(x)

		return x


class ContrastSourceNet_16_Skip(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_16_Skip, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 2*V, out_channels = 2*V*5, kernel_size = (3, 3), padding = 1)
		self.conv2 = nn.Conv2d(in_channels = 2*V*5, out_channels = 2*V*2, kernel_size = (5, 5), padding = 2)
		self.conv3 = nn.Conv2d(in_channels = 2*V*2, out_channels = 2*V, kernel_size = (3, 3), padding = 1)

	def forward(self, x):
		x = x + self.conv3( F.relu( self.conv2( F.relu( self.conv1( x ) ) ) ) )
		return x


class ContrastSourceNet_16_MultiScale(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_16_MultiScale, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels = 2*V, out_channels = 4*V, kernel_size = (3, 3), padding = 1)
		self.conv1_2 = nn.Conv2d(in_channels = 2*V, out_channels = 4*V, kernel_size = (5, 5), padding = 2)
		self.conv1_3 = nn.Conv2d(in_channels = 2*V, out_channels = 4*V, kernel_size = (7, 7), padding = 3)

		self.conv2 = nn.Conv2d(in_channels = 12*V, out_channels = 12*V, kernel_size = (3, 3), padding = 1)

		self.conv3_1 = nn.Conv2d(in_channels = 12*V, out_channels = 2*V, kernel_size = (3, 3), padding = 1)
		self.conv3_2 = nn.Conv2d(in_channels = 12*V, out_channels = 2*V, kernel_size = (5, 5), padding = 2)
		self.conv3_3 = nn.Conv2d(in_channels = 12*V, out_channels = 2*V, kernel_size = (7, 7), padding = 3)		

	def forward(self, x):
		x = F.relu(torch.cat( (self.conv1_1(x),self.conv1_2(x),self.conv1_3(x)), 1) )
		x = F.relu(self.conv2(x))
		x = self.conv3_1(x) + self.conv3_2(x) + self.conv3_3(x)

		return x



class ContrastSourceNet_16_MultiScale_1(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_16_MultiScale_1, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (3, 3), padding = 1)
		self.conv1_2 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (5, 5), padding = 2)
		self.conv1_3 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (7, 7), padding = 3)
		self.conv1_4 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (9, 9), padding = 4)

		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (5, 5), padding = 2)
		self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (5, 5), padding = 2)

		self.conv4_1 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (3, 3), padding = 1)
		self.conv4_2 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (5, 5), padding = 2)
		self.conv4_3 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (7, 7), padding = 3)
		self.conv4_4 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (9, 9), padding = 4)		

	def forward(self, x):
		x = F.relu(torch.cat( (self.conv1_1(x),self.conv1_2(x),self.conv1_3(x),self.conv1_4(x)), 1) )
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.conv4_1(x) + self.conv4_2(x) + self.conv4_3(x) + self.conv4_4(x)

		return x


class ContrastSourceNet_16_MultiScale_2(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_16_MultiScale_2, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (3, 3), padding = 1)
		self.conv1_2 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (5, 5), padding = 2)
		self.conv1_3 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (7, 7), padding = 3)
		self.conv1_4 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (9, 9), padding = 4)

		self.fc2 = nn.Linear(32*16*16,8*16*16)
		self.fc3 = nn.Linear(8*16*16,8*16*16)
		self.fc4 = nn.Linear(8*16*16,32*16*16)

		self.conv5_1 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (3, 3), padding = 1)
		self.conv5_2 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (5, 5), padding = 2)
		self.conv5_3 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (7, 7), padding = 3)
		self.conv5_4 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (9, 9), padding = 4)		

	def forward(self, x):
		x = F.relu(torch.cat( (self.conv1_1(x),self.conv1_2(x),self.conv1_3(x),self.conv1_4(x)), 1) )
		
		x = x.view(-1,32*16*16)
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = x.view(-1,32,16,16)

		x = self.conv5_1(x) + self.conv5_2(x) + self.conv5_3(x) + self.conv5_4(x)

		return x


class ContrastSourceNet_24_MultiScale_2(nn.Module):
	def __init__(self, V):
		super(ContrastSourceNet_24_MultiScale_2, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (3, 3), padding = 1)
		self.conv1_2 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (5, 5), padding = 2)
		self.conv1_3 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (7, 7), padding = 3)
		self.conv1_4 = nn.Conv2d(in_channels = 2*V, out_channels = 8, kernel_size = (9, 9), padding = 4)

		self.fc2 = nn.Linear(32*24*24,8*24*24)
		self.fc3 = nn.Linear(8*24*24,8*24*24)
		self.fc4 = nn.Linear(8*24*24,32*24*24)

		self.conv5_1 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (3, 3), padding = 1)
		self.conv5_2 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (5, 5), padding = 2)
		self.conv5_3 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (7, 7), padding = 3)
		self.conv5_4 = nn.Conv2d(in_channels = 32, out_channels = 2*V, kernel_size = (9, 9), padding = 4)		

	def forward(self, x):
		x = F.relu(torch.cat( (self.conv1_1(x),self.conv1_2(x),self.conv1_3(x),self.conv1_4(x)), 1) )
		
		x = x.view(-1,32*24*24)
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = x.view(-1,32,24,24)

		x = self.conv5_1(x) + self.conv5_2(x) + self.conv5_3(x) + self.conv5_4(x)

		return x