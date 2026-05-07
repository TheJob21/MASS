import numpy as np
import matplotlib.pyplot as plt

def HOCAE(frame, window_size, k, Pfa):
    '''
    Implementation of the HO-CAE algorithm for spectrum detections.

    frame: the frame for the threshold to be calculated on.
    WindowSize: estimator size. This should be a power of 2.
    k: order statistic for estimate selection.
    Pfa: Pfa used for threshold calculation.
    '''
    frame = np.asarray(frame)

    # Calculate the alpha value
    alpha = window_size * (Pfa ** (-1 / window_size) - 1)

    # Estimate the noise floor using overlapping windows
    estimates = []
    lower = 0
    upper = window_size
    step = window_size // 2

    while upper <= len(frame):
        tot = np.sum(frame[lower:upper])
        estimates.append(tot / window_size)
        lower += step
        upper += step

    estimate = np.array(estimates)

    # Select the order statistic (convert MATLAB's 1-based indexing to Python)
    sorted_estimate = np.sort(estimate)
    thresh = alpha * sorted_estimate[k - 1]

    return thresh, estimate

if __name__ == '__main__':
    # Load data and create spectrum snapshots
    filename = '../Data/union_spectrum_264ghz.dat'
    num_snapshots = 10000
    fft_size = 1024
    x = np.fromfile(filename, dtype=np.complex64, count=fft_size * num_snapshots)
    x = x.reshape((-1, fft_size)) # Time-domain snapshots
    X = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1) # Frequency domain
    
    # Plot spectrum
    X_db = 20*np.log10(abs(X) + 1e-10) # Complex to db
    plt.figure()
    plt.imshow(X_db, aspect='auto', origin='lower')
    plt.clim([X_db.max() - 50, X_db.max()])
    plt.colorbar()

    # HOCAE detector
    X_hocae = np.zeros(X.shape, dtype=bool)
    for i in range(num_snapshots):
        thresh, estimate =  HOCAE(abs(X[i]), window_size=32, k=5, Pfa=.0175)
        X_hocae[i] = abs(X[i]) > thresh
    plt.figure()
    plt.imshow(X_hocae, aspect='auto', origin='lower')
    plt.show()
