import numpy as np
import librosa
import scipy.signal as signal

def compute_sm_from_audio(x, L=21, H=5, L_smooth=16, tempo_rel_set=np.array([1]),
							 shift_set=np.array([0]), strategy='relative', scale=True, thresh=0.15,
							 penalty=0.0, binarize=False):
	# Waveform
	Fs = 22050
	x_duration = x.shape[0] / Fs

	# Chroma Feature Sequence and SSM (10 Hz)
	C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
	Fs_C = Fs / 2205

	# Chroma Feature Sequence and SSM
	X, Fs_feature = smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
	X = normalize_feature_sequence(X, norm='2', threshold=0.001)

	# Compute SSM
	S, I = compute_sm_ti(X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
	S_thresh = threshold_matrix(S, thresh=thresh, strategy=strategy,
										  scale=scale, penalty=penalty, binarize=binarize)
	return x, x_duration, X, Fs_feature, S_thresh, I

def smooth_downsample_feature_sequence(X, Fs, beats, filt_len=41, down_sampling=10, w_type='boxcar'):

    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    beats = beats[::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, beats, Fs_feature


def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):

    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

def compute_sm_ti(X, L=5, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2, thresh=None):

    for shift in shift_set:
        Y_cyc = shift_cyc_matrix(X, shift)
        S_cyc = np.dot(np.transpose(X), Y_cyc)

        if direction == 0:
            S_cyc = filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=0)
        if direction == 1:
            S_cyc = filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=1)
        if direction == 2:
            S_forward = filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift == shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0], S_cyc.shape[1])) * shift
        else:
            I_TI[S_cyc > S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)


    return S_TI


def shift_cyc_matrix(X, shift=0):
    X_cyc = np.roll(X, shift=shift, axis=0) 
    return X_cyc

def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal.
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        direction (int): Direction of smoothing (0: forward; 1: backward) (Default value = 0)

    Returns:
        S_L_final (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M / tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final


def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0.0, binarize=False):
    """Treshold matrix in a relative fashion

    Notebook: C4/C4S2_SSM-Thresholding.ipynb

    Args:
        S (np.ndarray): Input matrix
        thresh (float or list): Treshold (meaning depends on strategy)
        strategy (str): Thresholding strategy ('absolute', 'relative', 'local') (Default value = 'absolute')
        scale (bool): If scale=True, then scaling of positive values to range [0,1] (Default value = False)
        penalty (float): Set values below treshold to value specified (Default value = 0.0)
        binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)

    Returns:
        S_thresh (np.ndarray): Thresholded matrix
    """
    if np.min(S) < 0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N * M

    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N, M])

    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N, M])
        num_cells_row_below_thresh = int(np.round(M * (1-thresh_rel_row)))
        for n in range(N):
            row = S[n, :]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n, :] = (row >= thresh_abs)
        S_binary_col = np.zeros([N, M])
        num_cells_col_below_thresh = int(np.round(N * (1-thresh_rel_col)))
        for m in range(M):
            col = S[:, m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:, m] = (col >= thresh_abs)
        S_thresh = S * S_binary_row * S_binary_col

    if scale:
        cell_val_zero = np.where(S_thresh == 0)
        cell_val_pos = np.where(S_thresh > 0)
        if len(cell_val_pos[0]) == 0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])
        max_value = np.max(S_thresh)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value), (max_value - min_value))
            if len(cell_val_zero[0]) > 0:
                S_thresh[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        S_thresh[S_thresh > 0] = 1
        S_thresh[S_thresh < 0] = 0
    return S_thresh

def compute_scape_plot(S, fitness):
    low = S.shape[0] - len(fitness[0]) + 1

    scape = np.zeros((S.shape[0],S.shape[1]))
    for iy in range(len(fitness)):
        if iy < low:
            continue
        pad = S.shape[0] - len(fitness[iy])
        scape[iy,: -int(np.ceil(pad))] = fitness[iy].detach().cpu().numpy()

    return scape

def extract_chorus(scape, n_points=1):
    sorted_rev_idx = np.argsort(scape.ravel())
    row, col = np.unravel_index(sorted_rev_idx[-n_points:], scape.shape)
    top_n_idxs = list(zip(row, col))
    return top_n_idxs

def extract_chorus_segments(S, scape, tempo_deviation=0.1, similarity_threshold=0.05):
    segments = []
    start = None
    l, s  = extract_chorus(scape)[-1]
    sim_plot = np.sum(S[:, s: s+l + 1] + 1, axis=1)
    sim_plot[sim_plot < similarity_threshold * max(sim_plot)] = 0

    for i, value in enumerate(sim_plot):
        if value != 0:
            if start is None:
                start = i
        elif start is not None:
            segments.append((start, i - 1))
            start = None

    # If the last segment ends at the end of the array, add it
    if start is not None:
        segments.append((start, len(sim_plot) - 1))

    # Removing redundant primary chorus segment
    segments = [segment for segment in segments if np.argmax(sim_plot) in range(segment[0], segment[1] + 1)]

    # Filtering and combining segments
    filtered_segments = []
    prev_segment = segments[0]
    s_lims = [l*(1-tempo_deviation), l*(1+tempo_deviation)]

    for segment in segments[1:]:
        start_prev, end_prev = prev_segment
        start_current, end_current = segment

        if start_current - end_prev <= s_lims[0] and (end_current - start_prev <= s_lims[1]):
            # Combine the segments
            prev_segment = (start_prev, max(end_prev, end_current))
        else:
            # Add the previous segment to the result and update prev_segment
            filtered_segments.append(prev_segment)
            prev_segment = segment

    # Add the last segment to the result
    filtered_segments.append(prev_segment)

    # Filtering segments which are outside reasonable limits
    filtered_segments = [segment for segment in filtered_segments if segment[1] - segment[0] >= s_lims[0]]
    # Appending primary segment
    filtered_segments.append((s, s + l))


    return filtered_segments
