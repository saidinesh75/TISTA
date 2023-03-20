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
torch.set_default_dtype(torch.float64)

dir_name = "/home/saidinesh/TISTA/TISTA_figures/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

# device
device = torch.device('cuda:1') # choose 'cpu' or 'cuda'

# global variables
batch_size = 500  # mini-batch size
num_batch = 10  # number of mini-batches in a generation
num_generations = 1  # number of generations
# snr = 40.0  # SNR for the system in dB

# alpha2 = 1.0  # variance of non-zero componen
# alpha_std = math.sqrt(alpha2)
max_layers = num_generations  # maximum number of layers
adam_lr = 0.04  # initial learning parameter for Adam

''
# Loading the MUB Matrix
# data=loadmat("/home/saidinesh/Modulated_SPARCs/MUB_2_6.mat")
# A_ = np.array(data['B'])
# n,N = np.shape(A_)  # (64*4160)
# L = int(4)
# M = int(N/L)
''

code_params = {'P': 1,    # Average codeword symbol power constraint
               'R':0.5,
               'L': 4,    # Number of sections
               'M': 1024,      # Columns per section
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

# Just for the computation to work
# alpha2  =  1.1
# p = 1/M

# Generating the codebook for MAP detection after the last layer
# codeboook =  torch.eye(M)

# Generating the measurement Matrix
A = torch.normal(0.0, std=math.sqrt(P/L) * torch.ones(n, N))
# A_ = np.sqrt(n*P/L)*A_ 
# A = torch.from_numpy(A_)

At = A.t()
W = At.mm((A.mm(At)).inverse())  # pseudo inverse matrix
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
        # exp(89) is max for torch.float32 and exp(709) is max for torch.float64
        beta_th = torch.zeros(list(beta_hat.size()))
        for i in range(batch_size):
            s1 = beta_hat[:,i].clone()
            for j in range(L):
                beta_section = s1[int(delim[0,j]):int(delim[1,j]+1)]
                beta_th_section = torch.zeros(int(M))
                exp_param = (beta_section* torch.sqrt(n*P_vec[j]))/tau2[0,i]

                #initialization 
                new_exp_param = exp_param  # same will be used the max is less than 308
                max_exp_param = torch.max(new_exp_param)
                max_minus_term = 0
                if max_exp_param>709:
                    max_minus_term = max_exp_param - 709
                    new_exp_param = exp_param - max_minus_term
                denom = torch.sum(torch.exp(new_exp_param) ) 
                for k in range(int(M)):
                     num = torch.exp(  ((beta_section[k]* torch.sqrt(n*P_vec[j]))/tau2[0,i])  - max_minus_term )
                     beta_th_section[k] = num/denom
                beta_th[int(delim[0,j]):int(delim[1,j]+1), i] = beta_th_section  

        return beta_th
    '''
    
    ''
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
    ''
    
    '''
     def gauss(self, x,  var):
        return torch.exp(-torch.mul(x, x)/(2.0*var))/pow(2.0*math.pi*var,0.5)

    def MMSE_shrinkage(self, y, tau2):  # MMSE shrinkage function
        return (y*alpha2/(alpha2+tau2))*p*self.gauss(y,(alpha2+tau2))/((1-p)*self.gauss(y, tau2) + p*self.gauss(y, (alpha2+tau2)))

    def MMSE_shrinkage(self, y, tau2):
        temp0 = alpha2 + tau2
        temp1 = y*alpha2/temp0
        temp2 = temp1*p*self.gauss(y,(alpha2+tau2))

        temp3 = (1-p)*self.gauss(y, tau2)
        temp4 = p*self.gauss(y, (alpha2+tau2))

        return temp2/(temp3 + temp4)
    '''
    
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

# incremental training loop
start = time.time()

for gen in (range(num_generations)):
    # training process
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
    tot = 20 # batch size for accuracy check
    for i in range(tot):
        x = torch.Tensor(generate_msg_tista(code_params, batch_size)).to(device)
        x_hat = network(x, s_zero, gen+1).to(device)

        # MAP estimator
        for k in range(tot):
            for l in range(L):
                x_hat[l*M + torch.argmax( x_hat[int(delim[0,l]):int(delim[0,l]+1), k]) , k] = 1

        num = (x - x_hat).norm(2, 0).pow(2.0)
        denom = x.norm(2,0).pow(2.0)
        nmse = num/denom
        nmse_sum += torch.sum(nmse).item()

        sec_errors = torch.sum(torch.abs(x - x_hat))/2 
        sec_error = sec_errors/(batch_size*L)
        sec_err_sum += sec_error

    avg_sec_err = sec_err_sum / tot
    nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]

    print('({0}) NMSE= {1:6.3f} and avg_sec_err_rate = {a}'.format(gen + 1, nmse, a = avg_sec_err))
    # end of accuracy check    

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

torch.save(network.state_dict(),"/home/saidinesh/TISTA/trained_models/TISTA_SPARC_L2_")
print("Done")