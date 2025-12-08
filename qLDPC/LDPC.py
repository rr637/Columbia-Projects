import numpy as np
import matplotlib.pyplot as plt
from Plotting import *  # Assuming Plotting contains a function to plot results
from datetime import datetime

# Base protograph for the H matrix. Each element in this matrix represents a (e, e) sub-matrix:
# -1 represents an all-zero matrix
# b >= 0 represents an identity matrix with columns shifted by b positions.
# B is constructed to satisfy the regular (3,6) LDPC code conditions with rate R = 1/2.
# Here, each row has exactly 6 non-zero elements and each column has 3 non-zero elements.
# H construction and decoding are based on instructional videos from https://www.youtube.com/@nptel-nociitm9240

e = 40  # Size of sub-matrices in H
B = np.array([[0, -1, 1, 2, 3, -1, 4, 5, -1, -1],
              [-1, 0, 1, -1, -1, 2, 3, -1, 4, 5],
              [0, 1, -1, 2, 3, 4, -1, 5, -1, -1],
              [-1, 0, 1, -1, 2, -1, 3, -1, 4, 5],
              [0, -1, -1, 1, -1, 2, -1, 3, 4, 5]])

Sr, Sc = 6, 3  # Regularity parameters (3,6) for LDPC
Br, Bc = B.shape  # Shape of base protograph (5, 10)

k = (Bc - Br) * e  # Number of information bits, here 200
n = Bc * e  # Total codeword length, here 400
R = k / n  # Code rate (1/2)

# Initialize all-one message and all-zero codeword
message = np.ones(k)
cw = np.ones(n)

# Function to cyclically shift elements in an array by k positions
def shift(arr, k):
    n = arr.size
    k = n % k  
    if k == 0:
        return arr
    return np.concatenate((arr[-k:], arr[:-k]))

# LDPC Decoder Function
def LDPC_Decoder(EbNo_Db, Num_Error_Blocks, iterations):
    # Calculate Eb/No in linear scale and standard deviation for noise
    EbNo = 10 ** (EbNo_Db / 10)
    mean = 0
    sigma = 1 / np.sqrt(2 * R * EbNo)
    print(f'Sigma: {sigma}')
    print(f'EbNoDb: {EbNo_Db}')
    
    # Initialize error counters
    error_blocks = 0
    error_bits = 0
    Num_Blocks = 0
    
    # Perform decoding until the target number of error blocks is reached
    while error_blocks <= Num_Error_Blocks:
        # Add Gaussian noise to the all-zero codeword (cw) for each block
        y = cw + np.random.normal(mean, sigma, cw.shape)
        
        # Initialize storage matrices
        S = np.zeros((Sr * Br, e))  # Store non-zero elements in B
        Stemp = np.zeros((Sr, e))   # Temporary row storage during processing
        L = y  # Initialize LLRs to received noisy codeword
        
        i = 0
        while i < iterations:
            # Initialize counter for storage matrix S index
            Si = 0
            for row in range(Br):
                ti = 0
                for col in range(Bc):
                    if B[row, col] != -1:
                        start_index = col * e
                        end_index = (col + 1) * e
                        
                        # Update LLR by removing previous values from S
                        L[start_index:end_index] -= S[Si, :]
                        
                        # Shift LLRs and store them in Stemp
                        Stemp[ti, :] = shift(L[start_index:end_index], B[row, col])
                        ti += 1
                        Si += 1
                
                # Minsum algorithm processing for each in each row
                for j in range(e):
                    Stemp_sorted = np.unique(np.abs(Stemp[:ti, j]))
                    pos = np.argmin(Stemp[:ti, j])
                    min1, min2 = Stemp_sorted[0], Stemp_sorted[1]
                    sign = np.sign(Stemp[:ti, j])
                    parity = np.prod(sign)
                    
                    Stemp[:ti, j] = min1
                    Stemp[pos, j] = min2
                    Stemp[:ti, j] *= parity * sign
                
                # Restore counter values after row processing
                Si -= ti
                ti = 0
                for col in range(Bc):
                    if B[row, col] != -1:
                        #Reverse shift
                        S[Si, :] = shift(Stemp[ti, :], e - B[row, col])
                        start_index = col * e
                        end_index = (col + 1) * e
                        
                        # Update LLR by adding values from S
                        L[start_index:end_index] += S[Si, :]
                        
                        ti += 1
                        Si += 1
            
            # Convert LLRs to hard decisions and check for decoding success
            y_hat = np.where(L > 0, 1, np.where(L < 0, -1, 0))
            if np.all(y_hat == 1):
                break
            i += 1
        
        # Count errors in current block
        print(f'Block Number: {Num_Blocks}, Decoded Successfully: {np.all(y_hat == 1)}')
        print(f'(Errors, Iterations): {np.sum(y_hat != 1)}, {i}')
        error_bits += np.sum(y_hat != 1)
        Num_Blocks += 1
        if not np.all(y_hat == 1):
            error_blocks += 1
    
    # Compute Block Error Rate (BLER) and Bit Error Rate (BER)
    BLER = error_blocks / Num_Blocks
    BER = error_bits / (Num_Blocks * n)
    print(f"Error Blocks, Total Blocks: {(error_blocks, Num_Blocks)}")
    print(f"BLER: {BLER}")
    return BLER, BER

# Function to run the decoding experiment over an Eb/No range
def run_exp():
    # Experiment parameters
    SNL_Steps = 20
    Num_Error_Blocks = 50
    iterations = 16

    # Eb/No range in dB (SNL_Steps points from 0 to 4 dB)
    EbNo_Db_range = np.linspace(0, 3, SNL_Steps)
    
    BLER_results = []
    BER_results = []
    
    # Run decoder for each Eb/No point and collect results
    for EbNo_Db in EbNo_Db_range:
        BLER, BER = LDPC_Decoder(EbNo_Db, Num_Error_Blocks, iterations)
        BLER_results.append(BLER)
        BER_results.append(BER)
    
    # Save results to a file
    BLER_results = np.array(BLER_results)
    BER_results = np.array(BER_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Snl_Steps-{SNL_Steps}_NEBlocks-{Num_Error_Blocks}_Iter-{iterations}_{timestamp}'
    np.savez(filename, 
             EbNo_Db_range=EbNo_Db_range, 
             BLER_results=BLER_results,
             BER_results=BER_results)
    return filename

if __name__ == "__main__":
    filename = run_exp()
    plot_BLER_BER_results(filename)  #  function in Plotting to plot results
