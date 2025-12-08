import numpy as np
import matplotlib.pyplot as plt



def plot_BLER_BER_results(filename):
    # Load data from the saved file
    data = np.load(filename + '.npz')
    print(data.files)  # Check what data is stored in the file
    
    EbNo_Db_range = data['EbNo_Db_range']
    BLER_results = data['BLER_results']
    # BER_results = data['BER_results']  # Assuming BER_results is also saved in the file

    #Plotting BLER
    plt.figure(figsize=(10, 6))
    plt.semilogy(EbNo_Db_range, BLER_results, marker='o', linestyle='-', label="BLER")
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Block Error Rate (BLER)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(min(EbNo_Db_range), max(EbNo_Db_range))
    plt.ylim(1e-3, 1)  
    plt.xticks(np.arange(0, 3, 0.5))
    plt.title('BLER vs $E_b/N_0$')
    plt.legend()
    # plt.savefig(filename + "_BLER")
    plt.show()
    plt.close()

    # Plotting BER on a semilogarithmic scale
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(EbNo_Db_range, BER_results, marker='o', linestyle='-', label="BER")
    # plt.xlabel('$E_b/N_0$ (dB)')
    # plt.ylabel('Bit Error Rate (BER)')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid on both major and minor ticks
    # plt.xlim(min(EbNo_Db_range), max(EbNo_Db_range))
    # plt.title('BER vs $E_b/N_0$')
    # plt.legend()
    # plt.savefig(filename + "_BER")
    # plt.show()
    # plt.close()
    
  
# Filename (-.npz)
filename_manual = 'Snl_Steps-20_NEBlocks-50_Iter-16_20241107_135134'
if __name__ == "__main__":
    plot_BLER_BER_results(filename_manual)