import torch
import numpy as np


"""
All functions must retrieve the output as a numpy array.
"""


def post_process_peaks_binary(output, threshold=0.5, filter_peaks=False, min_distance=40):
    """
    Post-process the binary output of a model to identify peaks in the signal.
    Includes optional filtering to handle peaks that are too close together.

    :param output: Raw model output (logits or probabilities).
    :param threshold: Threshold for classifying a peak.
    :param filter_peaks: Whether to apply non-maximum suppression for closely spaced peaks.
    :param min_distance: Minimum distance (in samples) between peaks when filter_peaks=True.

    :return: Peak_indices (np.ndarray): 1D array with indices of detected peaks.
    """
    output_probs = torch.sigmoid(output).squeeze()
    if output_probs.dim != 1:
        # Ensure output_probs is 1D
        output_probs = output_probs.view(-1)

    output_binary = (output_probs > threshold).float()
    # Get all peak indices as 1D tensor
    all_peak_indices = torch.nonzero(output_binary, as_tuple=False).view(-1)

    if all_peak_indices.numel() == 0:
        print("No peaks were found.")
        return np.array([], dtype=int) 
    
    if not filter_peaks:
        return all_peak_indices.cpu().numpy()
    
    # Non-maximum suppression for closely spaced peaks
    all_peak_indices_np = all_peak_indices.cpu().numpy()

    if len(all_peak_indices_np) == 1:
        # Only one peak, nothing to filter
        return all_peak_indices_np

    peak_differences = np.diff(all_peak_indices_np)

    # If no peaks are closer than min_distance, return all
    if not np.any(peak_differences < min_distance):
        return all_peak_indices_np

    # Otherwise NMS
    filtered_peak_indices = []
    i = 0
    while i < len(all_peak_indices_np):
        window_start = all_peak_indices_np[i]
        window_end = window_start + min_distance

        # Peaks inside current window
        window_mask = (all_peak_indices_np >= window_start) & (all_peak_indices_np < window_end)
        window_peaks = all_peak_indices_np[window_mask]

        if len(window_peaks) > 0:
            # Keep the peak with highest probability in this window
            probs_np = output_probs[window_peaks].detach().cpu().numpy()
            max_peak = window_peaks[np.argmax(probs_np)]
            filtered_peak_indices.append(max_peak)

        i += len(window_peaks)

    return np.array(filtered_peak_indices, dtype=int)