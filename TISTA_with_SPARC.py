import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
torch.manual_seed(5)
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures 
rng = np.random.RandomState(seed=None)
from generate_msg_tista import generate_msg_tista

dir_name = "/home/saidinesh/TISTA/TISTA_figures/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

# device
device = torch.device('cuda:1') # choose 'cpu' or 'cuda'

# global variables
batch_size = 1000  # mini-batch size
num_batch = 200  # number of mini-batches in a generation
num_generations = 12  # number of generations
snr = 40.0  # SNR for the system in dB

alpha2 = 1.0  # variance of non-zero component
alpha_std = math.sqrt(alpha2)
max_layers = 12  # maximum number of layers
adam_lr = 0.04  # initial learning parameter for Adam

'''
# Loading the MUB Matrix
data=loadmat("/home/saidinesh/Modulated_SPARCs/MUB_2_6.mat")
A = np.array(data['B'])
n,_ = np.shape(A)  # (64*4160)
N = n**2
'''

code_params = {'P': 1,    # Average codeword symbol power constraint
               'R':0.5,
               'L': 4,    # Number of sections
               'M': 256,      # Columns per section
               'dist':0,
               'EbN0_dB':5,
               'modulated':False,
               'power_allocated':True,
               'spatially_coupled':False,
               'dist':0,
               'K':4
                }

code_params_list = ['P','R','L','M','EbN0_dB']
P,R,L,M,EbN0_dB = map(code_params.get, code_params_list)
N = int(L*M)
Eb_No_linear = np.power(10, np.divide(EbN0_dB,10))

# Calculating the length of the codeword and actual Rate
bit_len = int(round(L*math.log2(M)))
logM = int(round(math.log2(M)))
sec_size = logM
L = bit_len // sec_size
n = int(round(bit_len/R))
R_actual = bit_len/n
code_params.update({'R':R_actual})

# Generating the measurement Matrix
A = torch.normal(0.0, std=math.sqrt(n*P/L) * torch.ones(n, N)) 

At = A.t()
W = At.mm((A.mm(At)).inverse())  # pseudo inverse matrix
Wt = W.t()

taa = (At.mm(A)).trace().to(device)  # trace(A^T A)
tww = (W.mm(Wt)).trace().to(device)  # trace(W W^T)

Wt = torch.Tensor(Wt).to(device)
At = torch.Tensor(At).to(device)
x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device) 


# detection for NaN
def isnan(x):
    return x != x

class TISTA_SPARC_NET(nn.Module):
    def __init__(self):
        super(TISTA_SPARC_NET, self).__init__() 
        self.gamma = nn.Parameter(torch.ones(max_layers)) #nn.Parameter(torch.normal(1.0, 0.1*torch.ones(max_layers))) 
        print("TISTA initialized...")

    def MMSE_TISTA_SPARC(self,beta_hat,P_vec,tau2):
        # We are considering flat power distribution
        beta_th = np.zeros(torch.shape(beta_hat))
        rows,cols = beta_hat.shape 
        L = torch.size(P_vec)
        M = rows/L  

        # for i in range(cols):
        #     s1 = beta_hat[:,i].clone()
        #     for j in range(L):

    def eval_tau2(self, t, i):  # error variance estimator
        v2 = (t.norm(2,1).pow(2.0) - M*sigma2)/taa
        v2.clamp(min=1e-9)
        tau2 = (v2/N)*(N+(self.gamma[i]*self.gamma[i]-2.0*self.gamma[i])*M)+self.gamma[i]*self.gamma[i]*tww*sigma2/N
        tau2 = (tau2.expand(N, batch_size)).t()
        return tau2
        
    def forward(self, x, s, max_itr):  # TISTA network
        y = x.mm(At) + torch.Tensor(torch.normal(0.0, sigma_std*torch.ones(batch_size, M))).to(device)
        for i in range(max_itr):
            t = y - s.mm(At)
            tau2 = self.eval_tau2(t, i)
            r = s + t.mm(Wt)*self.gamma[i]
            s = self.MMSE_TISTA_SPARC(r, tau2)
        return s

global sigma_std, sigma2, xi  

network = TISTA_SPARC_NET().to(device)  # generating an instance of TISTA network
s_zero = torch.Tensor(torch.zeros(N, batch_size)).to(device)  # initial value
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)

# SNR calculation
Eb = n*P/bit_len
awgn_var = Eb/Eb_No_linear
sigma = np.sqrt(awgn_var)
code_params.update({'awgn_var':awgn_var})
snr_rx = P/awgn_var
capacity = 0.5 * np.log2(1 + snr_rx)

# incremental training loop
start = time.time()

for gen in (range(num_generations)):
    # training process
    for i in range(num_batch):
        if (gen > 10):
            opt = optim.Adam(network.parameters(), lr = adam_lr/50.0)
        x = torch.Tensor(generate_msg_tista).to(device)
        opt.zero_grad()
        x_hat = network(x, s_zero, x)
        loss = F.mse_loss(x_hat,x)
        loss.backward()

        grads = torch.stack([param.grad for param in network.parameters()])
        if isnan(grads).any():  # avoiding NaN in gradients
            continue
        opt.step()
        # end of training

    # accuracy check after t-th incremental training
    nmse_sum = 0.0
    tot = 1 # batch size for accuracy check
    for i in range(tot):
        x = torch.Tensor(generate_msg_tista()).to(device)
        x_hat = network(x, s_zero, gen+1).to(device)
        num = (x - x_hat).norm(2, 1).pow(2.0)
        denom = x.norm(2,1).pow(2.0)
        nmse = num/denom
        nmse_sum += torch.sum(nmse).item()

    nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]

    print('({0}) NMSE= {1:6.3f}'.format(gen + 1, nmse))
    # end of accuracy check    

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print("Done")