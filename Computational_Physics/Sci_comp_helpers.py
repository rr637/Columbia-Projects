import numpy as np

### PSET2

def kin_energy(vel):
    return 24.0 * np.sum(vel * vel)

def temp(vel,N):
    return (16.0 / N) * np.sum(vel * vel)

def r_ij_matrix(pos, L):
    Nloc = pos.shape[0]
    d = pos.reshape(Nloc, 1, 3) - pos.reshape(1, Nloc, 3)
    d -= L * np.around(d / L)  
    return d

def net_force(pos, L):
    d  = r_ij_matrix(pos, L)                # (N,N,3)
    r2 = np.sum(d * d, axis=2)              # (N,N)
    np.fill_diagonal(r2, np.inf)            # kill diagonal
    invr2   = 1.0 / r2
    f_scale = (invr2**7) - 0.5*(invr2**4)   # (N,N)
    F = np.sum(d * f_scale[:, :, None], axis=1)   # (N,3)
    return F

def pot_energy(pos, L):
    d  = r_ij_matrix(pos, L)
    r2 = np.sum(d * d, axis=2)
    np.fill_diagonal(r2, np.inf)
    iu = np.triu_indices(pos.shape[0], k=1)     # i<j
    invr2 = 1.0 / r2[iu]
    invr6  = invr2**3
    u_pairs = 4.0 * (invr6**2 - invr6)
    return np.sum(u_pairs)


def virial(pos, L):

    d  = r_ij_matrix(pos, L)           # (N,N,3)
    r2 = np.sum(d*d, axis=2)           # (N,N)
    np.fill_diagonal(r2, np.inf)

    iu = np.triu_indices(pos.shape[0], 1)  # i<j
    invr2 = 1.0 / r2[iu]
    invr6 = invr2**3
    g_pairs = (invr6**2) - 0.5*invr6
    return np.sum(g_pairs)

def pressure_from_virial(G, N, L, T):
    V = float(L)**3
    rho = N / V
    return rho * T + (G / (3.0 * V))


def unload_from_npz(path):

  with np.load(path, allow_pickle=True) as d:
      pos = d["pos"]         
      vel = d["vel"]          
      steps = d["steps"] 

    
      PEs, KEs, TEs = d["PEs"], d["KEs"], d["TEs"]
      Forces = d["Forces"]
  return pos, vel, steps, Forces,PEs, KEs,TEs

### PSET 3

def acf(X, ncut):
    """
    Returns normalized  rho ACF: C(n), shape (ncut+1, A).
    """
    M, A = X.shape
    ncut = min(int(ncut), M - 1)

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    C = np.empty((ncut + 1, A), dtype=np.float64)
    for n in range(ncut + 1):
        prod = Xc[:M - n, :] * Xc[n:, :]
        C[n, :] = prod.sum(axis=0) / (M - n)
    C0 = C[0, :]
    rho  = C/C0
    return rho,C,C0

def get_tau_int(rho,ncut):
  tau_ints = 0.5 + rho[1:ncut+1, :].sum(axis=0)
  return tau_ints

def norm_covariance(X):
  M,A = X.shape
  means = X.mean(axis=0)
  Xc = X - means
  C = (Xc.T @ Xc) / M 
  sigmas = np.sqrt(np.diag(C))      
  rho = C / np.outer(sigmas,sigmas)       
  return rho   

def bin_means(XN, b):
 
    N, A = XN.shape
    B = N // b
    X_trim = XN[:B*b]                   
    return X_trim.reshape(B, b, A).mean(axis=1)  # (B, A)

def jackknife__means(B_means):

    B, A = B_means.shape
    total = B_means.sum(axis=0)         # (A,)
    jk_vals = (total - B_means) / (B - 1)  # (B,A)
    return jk_vals