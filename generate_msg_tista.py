import numpy as np
import torch
import math

def is_power_of_2(x):
    return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation.

def gray2bin(num):
    '''
    Converts gray code (int type) to binary code (int type)
    From https://en.wikipedia.org/wiki/Gray_code
    '''
    mask = num >> 1
    while (mask != 0):
        num  = num ^ mask
        mask = mask >> 1
    return num

def bin_arr_2_int(bin_array):
    '''
    Binary array (numpy.ndarray) to integer
    '''
    # assert bin_array.dtype == 'bool'
    k = bin_array.size()
    assert 0 < k < 64 # Ensures non-negative integer output
    # 1 << torch.arange(k)[::-1]      generates vector like [16 8 4 2 1]
    return bin_array.dot(1 << torch.arange(k)[::-1])

def psk_constel(K):
    '''
    K-PSK constellation symbols
    '''
    assert type(K)==int and K>1 and is_power_of_2(K)

    if K == 2:
        c = torch.array([1, -1])
    elif K == 4:
        c = torch.array([1+0j, 0+1j, -1+0j, 0-1j])
    else:
        theta = 2*torch.pi*torch.arange(K)/K
        c     = torch.cos(theta) + 1J*torch.sin(theta)

    return c

def psk_mod(bin_arr, K):
    '''
    K-PSK modulation (using gray coding).

    bin_arr: boolean numpy.ndarray to modulate. Length of  L * log2(K).
    K      : number of PSK contellations, K>1 and is a power of 2

    Returns
    symbols: Corresponding K-PSK modulation symbols of length L.
             (If K=2 then symbols are real, complex otherwise.)
    '''

    assert type(K)==int and K>1 and is_power_of_2(K)
    # assert bin_arr.dtype == 'bool'

    c    = psk_constel(K)           # Constellation symbols
    logK = int(round(math.log2(K)))
    assert bin_arr.size() % logK == 0
    L    = bin_arr.size() // logK     # Number of symbols
    if L == 1:
        k = bin_arr.size()
        idx     = gray2bin(bin_arr_2_int(bin_arr)) # gray code index
        symbols = c[idx]
    else:
        symbols = torch.zeros(L, dtype=c.dtype)
        for l in range(L):
            idx        = gray2bin(bin_arr_2_int(bin_arr[l*logK:(l+1)*logK]))
            symbols[l] = c[idx]

    return symbols

def generate_msg_tista(code_params,cols):
    P,L,M,dist,awgn_var = map(code_params.get,['P','L','M','dist','awgn_var'])
    K = code_params['K'] if code_params['modulated'] else 1
    N = int(L*M)

    bit_len = int(round(L*math.log2(K*M)))
    logM = int(round(math.log2(M)))
    logK = int(round(math.log2(K)))

    if K==1: # unmodulated case
        sec_size = logM  # sec_seize = 5
    else:
        assert type(K)==int and K>1 and is_power_of_2(K)
        logK = int(round(math.log2(K)))
        sec_size = logM + logK

    beta = torch.zeros((N,cols),dtype=torch.cfloat) if K>2 else torch.zeros((N,cols))
    # beta2 = torch.zeros((N,cols))

    for i in range(cols):
        bits_in = torch.randint(2, size=(bit_len,))
        if K==1 or K==2:
            beta0 = torch.zeros((N,))    #length of msg_vector = 1000 * 32 = 32000
            # beta1 = torch.zeros(N)
        else:
            beta0 = torch.zeros((N,), dtype=torch.cfloat)
            # beta1 = torch.zeros(N, dtype=complex)

        for l in range(L):
            bits_sec = bits_in[l*sec_size : l*sec_size + logM]  # logM bits used for selection the location of non-zero values
            assert 0<logM<64
            idx = bits_sec.dot(1 << torch.flip(torch.arange(logM),[0,]))

            if K==1:
                val = 1
            else:
                bits_mod_sec = bits_in[l*sec_size+logM : (l+1)*sec_size] #logK bits used for selection of PSK symbol for the non-zero location
                val = psk_mod(bits_mod_sec, K)

            beta0[l*int(M) + idx] = val      # will make a 1 at the decimal equivalent in the l-th section
        
        beta[:,i] = beta0

    # c = psk_constel(K)    
    return beta   