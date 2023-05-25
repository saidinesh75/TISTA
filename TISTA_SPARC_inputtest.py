# Trainable ISTA (TISTA)
# 
# This code is an implementation of Trainable ISTA (TISTA) for sparse signal recovery in Pytorch.Tensor.
# The details of the algorithm can be found in the paper:
# Daisuke Ito, Satoshi Takabe, Tadashi Wadayama,
# "Trainable ISTA for Sparse Signal Recovery", arXiv:1801.01978.
# (Computer experiments in the paper was performed with another TensorFlow implementation)
#
# GPU is required for execution of this program. If you do not have GPU,
# just change "device = torch.device('cuda')" to 'cpu'.
# 
# This basic TISTA trains only $\gamma_t$.
#
# Last update 11/21/2018
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import time
torch.set_default_dtype(torch.float64)
from generate_msg_tista import generate_msg_tista

# device
device = torch.device('cuda') # choose 'cpu' or 'cuda'

# global variables

N = 512  # length of a source signal vector
n = 250  # length of a observation vector
p = 0.1  # probability for occurrence of non-zero components

batch_size = 1000  # mini-batch size
num_batch = 200  # number of mini-batches in a generation
num_generations = 12  # number of generations
snr = 40.0  # SNR for the system in dB

alpha2 = 1.0  # variance of non-zero component
alpha_std = math.sqrt(alpha2)
max_layers = 12  # maximum number of layers
adam_lr = 0.04  # initial learning parameter for Adam

learning=1
# random seed of torch
torch.manual_seed(5)

### setting sensing matrix
# sensing matrix with small variance
A = torch.normal(0.0, std=math.sqrt(1.0/n) * torch.ones(n, N)) 

#####... SPARC encoding.....####
code_params = {'P': 1,    # Average codeword symbol power constraint
                'L': 4,    # Number of sections
                'M': 128,      # Columns per section
                'dist':0,
                'n':n,
                'EbN0_dB':40,
                'modulated':False,
                'power_allocated':True,
                'spatially_coupled':False,
                'dist':0,
                'K':0
                }

P,L,M = map(code_params.get, ['P','L','M'])

delim = torch.zeros([2,L])
delim[0,0] = 0
delim[1,0] = M-1

for i in range(1,L):
    delim[0,i] = delim[1,i-1]+1
    delim[1,i] = delim[1,i-1]+M
# sensing matrix with large variance
#A = torch.normal(0.0, std=math.sqrt(1.0) * torch.ones(M, N))  

# \pm 1 sensing matrix 
#A = 1.0-2.0*torch.bernoulli(0.5 * torch.ones(M, N))  
### end of setting sensing matrix

At = A.t()
W = At.mm((A.mm(At)).inverse())  # pseudo inverse matrix
Wt = W.t()

taa = (At.mm(A)).trace().to(device)  # trace(A^T A)
tww = (W.mm(Wt)).trace().to(device)  # trace(W W^T)

W = torch.Tensor(W).to(device)
A = torch.Tensor(A).to(device)

# print("sensing matrix A\n", A.detach().numpy())

# detection for NaN
def isnan(x):
    return x != x

# mini-batch generator
# def generate_batch():
#     support = torch.bernoulli(p * torch.ones(N, batch_size))
#     nonzero = torch.normal(0.0, alpha_std * torch.ones(N, batch_size))
#     return torch.mul(nonzero, support)

# definition of TISTA network
class TISTA_NET(nn.Module): 
    def __init__(self):
        super(TISTA_NET, self).__init__() 
        self.gamma = nn.Parameter(torch.ones(max_layers)) #nn.Parameter(torch.normal(1.0, 0.1*torch.ones(max_layers))) 
        print("TISTA initialized...")
    
    def gauss(self, x,  var):
        return torch.exp(-torch.mul(x, x)/(2.0*var))/pow(2.0*math.pi*var,0.5)

    def MMSE_shrinkage(self, y, tau2):  # MMSE shrinkage function
        return (y*alpha2/(alpha2+tau2))*p*self.gauss(y,(alpha2+tau2))/((1-p)*self.gauss(y, tau2) + p*self.gauss(y, (alpha2+tau2)))

    def eval_tau2(self, t, i):  # error variance estimator
        v2 = (t.norm(2,0).pow(2.0) - n*sigma2)/taa
        v2.clamp(min=1e-9)
        tau2 = (v2/N)*(N+(self.gamma[i]*self.gamma[i]-2.0*self.gamma[i])*n)+self.gamma[i]*self.gamma[i]*tww*sigma2/N
        tau2 = (tau2.expand(N,batch_size))
        return tau2

    def forward(self, x, s, max_itr):  # TISTA network
        y = A.mm(x) + torch.Tensor(torch.normal(0.0, sigma_std*torch.ones(n,batch_size))).to(device)
        for i in range(max_itr):
            t = y - A.mm(s)
            tau2 = self.eval_tau2(t, i)
            r = s + W.mm(t)*self.gamma[i]
            s = self.MMSE_shrinkage(r, tau2)
        return s

global sigma_std, sigma2, xi

network = TISTA_NET().to(device)  # generating an instance of TISTA network
s_zero = torch.Tensor(torch.zeros(N,batch_size)).to(device)  # initial value
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)
network_path = "/home/dinesh/Research_work/TISTA/TISTA_trained_models_2/S_TISTA_n250N512_SNR40_L12" # gtdB_learnable means gamma, tau, delta and B are learnable

# SNR calculation
sum = 0.0
for i in range(100):
    x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device)
    y = A.mm(x)
    sum += (torch.norm(y, p=2).pow(2.0)).sum().item()
ave = sum/(100.0 * batch_size)
sigma2 = ave/(n*math.pow(10.0, snr/10.0))
sigma_std = math.sqrt(sigma2)
xi = alpha2 + sigma2

# incremental training loop
start = time.time()

for gen in (range(num_generations)):
    # training process 
    if learning:
        for i in range(num_batch):
            if os.path.exists(network_path):
                    network.load_state_dict(torch.load(network_path))
                    break
            if (gen > 10): # change learning rate of Adam
                opt = optim.Adam(network.parameters(), lr=adam_lr/50.0)
            x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device)
            opt.zero_grad()
            x_hat = network(x, s_zero, gen+1).to(device)
            loss = F.mse_loss(x_hat, x)
            loss.backward()

            grads = torch.stack([param.grad for param in network.parameters()])
            if isnan(grads).any():  # avoiding NaN in gradients
                continue
            opt.step()
    # end of training training

    # accuracy check after t-th incremental training
    nmse_sum = 0.0
    sec_err_sum = 0.0
    tot = 25 # batch size for accuracy check
    for i in range(tot):
        x = torch.Tensor(generate_msg_tista(code_params,batch_size)).to(device)
        x_hat = network(x, s_zero, gen+1).to(device)
        # num = (x - x_hat).norm(2, 1).pow(2.0)
        # denom = x.norm(2,1).pow(2.0)
        # nmse = num/denom
        # nmse_sum += torch.sum(nmse).item()

        # MAP estimator
        # for k in range(tot):
        for l in range(L):
            # x_hat[l*M + torch.argmax( x_hat[int(delim[0,l]):int(delim[0,l]+1), k]) , k] = 1
            index = torch.argmax(x_hat[int(delim[0,l]):int(delim[1,l]+1), :],dim=0)
            x_hat[int(delim[0,l]):int(delim[1,l]+1), :] =torch.eye(M)[:,index]

        sec_errors = torch.sum(torch.abs(x - x_hat))/2 
        sec_error = sec_errors/(batch_size*L)
        sec_err_sum = sec_err_sum + sec_error.item()

    # nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]
    avg_sec_err = sec_err_sum / tot

    # print('({0}) NMSE= {1:6.3f} and avg_sec_err_rate = {a}'.format(gen + 1, nmse,a=avg_sec_err))
    print('({0}) avg_sec_err_rate = {a}'.format(gen + 1,a=avg_sec_err))
    # end of accuracy check
 
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

torch.save(network.state_dict(),network_path)

print("Done")