import numpy as np

class SignalProcessor:
    def __init__(self, config):
        self.cfg = config

    def HOCAE(self, frame, window_size, k, Pfa):
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

    def fill_small_gaps(self, occupancy, max_gap=10):
        filled = occupancy.copy()
        n = len(occupancy)
        
        i = 0
        while i < n:
            if not occupancy[i]:
                start = i
                
                # find end of gap
                while i < n and not occupancy[i]:
                    i += 1
                end = i
                
                gap_size = end - start
                
                # check if bounded by True on both sides
                left = start - 1
                right = end
                
                if (
                    gap_size <= max_gap and
                    left >= 0 and right < n and
                    occupancy[left] and occupancy[right]
                ):
                    filled[start:end] = True
            else:
                i += 1

        return filled

    def compute_state_from_file(self, f):
        data = np.fromfile(f, dtype=np.complex64, count=self.cfg.FFT_SIZE)
        if data.size < self.cfg.FFT_SIZE:
            return None
        # FFT → frequency domain
        X = np.fft.fftshift(np.fft.fft(data))
        mag = np.abs(X)
        # HO-CAE detection
        thresh, _ = self.HOCAE(
            mag,
            window_size=32,
            k=self.cfg.HOCAE_ORDER_SELECTION,
            Pfa=1e-2
        )
        # Boolean occupancy state
        occupancy = mag > thresh

        occupancy = self.fill_small_gaps(occupancy=occupancy, max_gap=10)

        return occupancy