import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
# torch.manual_seed(5)
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
torch.set_default_dtype(torch.float64)

dir_name = "/home/dinesh/Research_work/TISTA/TISTA_figures/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

# device
device = torch.device('cuda:1') # choose 'cpu' or 'cuda'

# global variables
batch_size = 500  # mini-batch size
num_batch = 10  # number of mini-batches in a generation
num_generations = 5  # number of generations
max_layers = num_generations  # maximum number of layers
adam_lr = 0.04  # initial learning parameter for Adam
learning = 1

''' Loading MUB matrix
# Loading the MUB Matrix
# data=loadmat("/home/dinesh/Research_work/Dinesh_SPARC_codes_2/gold_mat_files/goldi_31")
# A_ = np.array(data['B'])
# n,N = np.shape(A_)  # (64*4160)
A = torch.from_numpy(A_)

# L = int(4)
# M = int(N/L)
'''

code_params = {'P': 1,    # Average codeword symbol power constraint
               'R':0.4375,  # 0.4375 will give n=64
               'L': 4,    # Number of sections
               'M': 128,      # Columns per section
               'dist':0,
               'EbN0_dB':15,
               'modulated':False,
               'power_allocated':True,
               'spatially_coupled':False,
               'dist':0,
               'K':0
                }

code_params_list = ['P','R','L','M','EbN0_dB']
P,R,L,M,EbN0_dB = map(code_params.get, code_params_list)
N = int(L*M)
P_vec = (P/L)*torch.ones(L)
Eb_No_linear = math.pow(10,(EbN0_dB/10))

# index calculation
delim = torch.zeros([2,L])
delim[0,0] = 0
delim[1,0] = M-1

for i in range(1,L):
    delim[0,i] = delim[1,i-1]+1
    delim[1,i] = delim[1,i-1]+M

# Calculating the length of the codeword and actual Rate
bit_len = int(round(L*math.log2(M)))
logM = int(round(math.log2(M)))
sec_size = logM
L = bit_len // sec_size
n = int(round(bit_len/R))
R_actual = bit_len/n
code_params.update({'R':R_actual})

# Generating the measurement Matrix
A = torch.normal(0.0, std=math.sqrt(P/L) * torch.ones(n, N))

# A = torch.from_numpy(A_)
# A_ = np.sqrt(n*P/L)*A_ 

At = A.t()
W = At.mm((A.mm(At)).inverse())  # pseudo inverse matrix  (Tried using Goldcode matrix but they are not invertible)
Wt = W.t()

taa = (At.mm(A)).trace().to(device)  # trace(A^T A)
tww = (W.mm(Wt)).trace().to(device)  # trace(W W^T)

W = torch.Tensor(W).to(device)
A = torch.Tensor(A).to(device)
# x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device) 

# detection for NaN
def isnan(x):
    return x != x

class TISTA_SPARC_NET(nn.Module):
    def __init__(self):
        super(TISTA_SPARC_NET, self).__init__() 
        self.gamma = nn.Parameter(torch.ones(max_layers)) #nn.Parameter(torch.normal(1.0, 0.1*torch.ones(max_layers))) 
        print("TISTA initialized...")

    '''
    def MMSE_sparc_shrinkage(self, beta_hat, tau2):
        beta_th = torch.zeros(list(beta_hat.size()))
        for i in range(batch_size):
            s1 = beta_hat[:,i]
            for j in range(L):
                exp_param = ( s1[int(delim[0,j]):int(delim[1,j]+1)] * torch.sqrt(n*P_vec[j]) )/tau2[0,i]
                max_exp_param = torch.max(exp_param)
                max_minus_term = 0
                if max_exp_param > 709:
                    max_minus_term = max_exp_param - 709
                    exp_param = exp_param - max_minus_term
                denom = torch.sum(torch.exp(exp_param))
                beta_th[int(delim[0,j]):int(delim[1,j]+1), i] = torch.exp(exp_param)/denom
    
        return beta_th
    '''
    
    def MMSE_sparc_shrinkage(self, beta_hat, tau2):
        beta_th = torch.zeros(list(beta_hat.size()))
        for j in range(L):
            exp_param = (beta_hat[int(delim[0,j]):int(delim[1,j]+1),:] * torch.sqrt(n*P_vec[j]))/tau2[0,:]
            max_exp_param = torch.max(exp_param)
            max_minus_term = 0
            if max_exp_param > 709:
                max_minus_term = max_exp_param - 709
                exp_param = exp_param - max_minus_term
            denom = torch.sum(torch.exp(exp_param),dim=0)
            beta_th[int(delim[0,j]):int(delim[1,j]+1), :] = torch.exp(exp_param)/denom
    
        return beta_th
    
    def eval_tau2(self, t, i):  # error variance estimator
        v2 = (t.norm(2,0).pow(2.0) - M*sigma2)/taa
        v2.clamp(min=1e-9)
        tau2 = (v2/N)*(N+(self.gamma[i]*self.gamma[i]-2.0*self.gamma[i])*M)+self.gamma[i]*self.gamma[i]*tww*sigma2/N
        tau2 = (tau2.expand(N, batch_size))
        return tau2
      
    def forward(self, x, s, max_itr):  # TISTA network
        y = A.mm(x) + torch.Tensor(torch.normal(0.0, sigma_std*torch.ones(n,batch_size))).to(device)
        for i in range(max_itr):
            t = y - A.mm(s)
            tau2 = self.eval_tau2(t, i)
            r = s + W.mm(t)*self.gamma[i]
            s = torch.Tensor(self.MMSE_sparc_shrinkage(r, tau2)).to(device)
            # s_new = torch.Tensor(self.MMSE_sparc_shrinkage_new(r, tau2)).to(device)
        return s
    
global sigma_std, sigma2, xi  

network = TISTA_SPARC_NET().to(device)  # generating an instance of TISTA network
s_zero = torch.Tensor(torch.zeros(N,batch_size)).to(device)  # initial value
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)

# SNR calculation
Eb = n*P/bit_len
sigma2 = Eb/Eb_No_linear
sigma_std = math.sqrt(sigma2)
code_params.update({'awgn_var':sigma2})
snr_rx = P/sigma2
capacity = 0.5 * math.log2(1 + snr_rx)

network_path = "/home/dinesh/Research_work/TISTA/TISTA_trained_models_new/ST_n{n}N{N}_E{ebno}_L{layers}".format(n=n, N=N,ebno = EbN0_dB,layers=max_layers) # gtdB_learnable means gamma, tau, delta and B are learnable

# incremental training loop
start = time.time()

for gen in (range(num_generations)):
    # training process
    if os.path.exists(network_path):
        network.load_state_dict(torch.load(network_path))
    else:
        for i in range(num_batch):
            if (gen > 10):
                opt = optim.Adam(network.parameters(), lr = adam_lr/50.0)
            x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device)
            opt.zero_grad()
            x_hat = network(x, s_zero, gen +1).to(device)
            
            loss = F.mse_loss(x_hat,x)
            loss.backward()

            grads = torch.stack([param.grad for param in network.parameters()])
            if isnan(grads).any():  # avoiding NaN in gradients
                continue
            opt.step()
        # end of training

    # accuracy check after t-th incremental training
    nmse_sum = 0.0
    sec_err_sum = 0.0
    tot = 25 # batch size for accuracy check
    for i in range(tot):
        x = torch.Tensor(generate_msg_tista(code_params, batch_size)).to(device)
        x_hat = network(x, s_zero, gen+1).to(device)

        # num = (x - x_hat).norm(2, 0).pow(2.0)
        # denom = x.norm(2,0).pow(2.0)
        # nmse = num/denom
        # nmse_sum += torch.sum(nmse).item()

        # MAP estimator
        for l in range(L):
            index = torch.argmax(x_hat[int(delim[0,l]):int(delim[1,l]+1), :],dim=0)
            x_hat[int(delim[0,l]):int(delim[1,l]+1), :] =torch.eye(M)[:,index]        

        sec_errors = torch.sum(torch.abs(x - x_hat))/2 
        sec_error = sec_errors/(batch_size*L)
        sec_err_sum = sec_err_sum + sec_error.item()

    avg_sec_err = sec_err_sum / tot
    # nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]

    # print('({0}) NMSE= {1:6.3f} and avg_sec_err_rate = {a}'.format(gen + 1, nmse, a = avg_sec_err))
    print('({0}) avg_sec_err_rate = {a}'.format(gen + 1, a = avg_sec_err))
    # end of accuracy check    

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

torch.save(network.state_dict(),network_path)
print("Done")