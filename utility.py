import numpy as np
import pywt

def p_omega(x, indices):
    """
        Definition of P_Omega
    """
    return np.expand_dims(x[np.unravel_index(indices, x.shape)], 1)

def p_omega_t(x, indices, m):
    """
        Definition of P_Omega^T
    """
    y = np.zeros((m, m))
    y[np.unravel_index(indices, y.shape)] = x.squeeze()
    return y

def l1_prox(y, weight):
    """
        Projection onto the L1 Norm Ball
    """
    return np.sign(y)*np.maximum(np.absolute(y) - weight, 0)

def tv_norm(X, type=None):
    """
        Computes the TV-norm of image X based on type ['iso' for isotropic and anisotropic by default]
    """

    m, n = X.shape
    P1 = X[0:m - 1, :] - X[1:m, :]
    P2 = X[:, 0:n - 1] - X[:, 1:n]

    if type == 'iso':
        D = np.zeros_like(X)
        D[0:m - 1, :] = P1 ** 2
        D[:, 0:n - 1] = D[:, 0:n - 1] + P2 ** 2
        tv_out = np.sum(np.sqrt(D))
    else:
        tv_out = np.sum(np.abs(P1)) + np.sum(np.abs(P2))

    return tv_out

class WaveletTransform(object):
    """
        Contains the forward and adjoint operators for the Wavelet Transform
    """
    def __init__(self, m=256):
        self.m = m
        self.N = m ** 2

        self.W_operator = lambda x: pywt.wavedec2(x, 'db8', mode='periodization')
        self.WT_operator = lambda x: pywt.waverec2(x, 'db8', mode='periodization')
        _, self.coeffs = pywt.coeffs_to_array(self.W_operator(np.ones((m, m))))

    def W(self, x):
        """
            Computes the Wavelet transform from a vectorized image.
        """
        x = np.reshape(x, (self.m, self.m))
        wav_x, _ = pywt.coeffs_to_array(self.W_operator(x))

        return np.reshape(wav_x, (self.N, 1))

    def WT(self, wav_x):
        """
            Computes the adjoint Wavelet transform from a vectorized image.
        """
        wav_x = np.reshape(wav_x, (self.m, self.m))
        x = self.WT_operator(pywt.array_to_coeffs(wav_x, self.coeffs, output_format='wavedec2'))
        return np.reshape(x, (self.N, 1))
