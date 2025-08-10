"""
Constant False Alarm Rate (CFAR) detection algorithms.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.special import gammaincinv


class CFARDetector:
    """Base class for CFAR detectors."""
    
    def __init__(self, guard_cells: int, training_cells: int, pfa: float):
        """
        Initialize CFAR detector.
        
        Args:
            guard_cells: Number of guard cells on each side
            training_cells: Number of training cells on each side
            pfa: Probability of false alarm
        """
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        
        # Total window size
        self.window_size = 2 * (guard_cells + training_cells) + 1
        
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Apply CFAR detection to data.
        
        Args:
            data: Input data (magnitude squared)
            
        Returns:
            Binary detection map
        """
        raise NotImplementedError("Subclasses must implement detect()")


class CellAveragingCFAR(CFARDetector):
    """Cell-Averaging CFAR (CA-CFAR) detector."""
    
    def __init__(self, guard_cells: int, training_cells: int, pfa: float):
        """Initialize CA-CFAR detector."""
        super().__init__(guard_cells, training_cells, pfa)
        
        # Calculate threshold multiplier based on PFA
        # For exponentially distributed noise (after square-law detector)
        self.threshold_factor = self._calculate_threshold_factor()
    
    def _calculate_threshold_factor(self) -> float:
        """Calculate threshold multiplier for desired PFA."""
        n = 2 * self.training_cells  # Total training cells
        return n * (self.pfa**(-1/n) - 1)
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Apply CA-CFAR detection.
        
        Args:
            data: Input data (magnitude squared)
            
        Returns:
            Binary detection map
        """
        detections = np.zeros_like(data, dtype=bool)
        
        # Pad data for edge handling
        pad_width = self.guard_cells + self.training_cells
        padded_data = np.pad(data, pad_width, mode='edge')
        
        # Apply CFAR for each cell
        for i in range(len(data)):
            # Get cell under test
            cut_idx = i + pad_width
            cut_value = padded_data[cut_idx]
            
            # Get training cells (excluding guard cells)
            left_start = cut_idx - pad_width
            left_end = cut_idx - self.guard_cells
            right_start = cut_idx + self.guard_cells + 1
            right_end = cut_idx + pad_width + 1
            
            training_cells = np.concatenate([
                padded_data[left_start:left_end],
                padded_data[right_start:right_end]
            ])
            
            # Calculate threshold
            noise_estimate = np.mean(training_cells)
            threshold = self.threshold_factor * noise_estimate
            
            # Detection decision
            detections[i] = cut_value > threshold
        
        return detections


class OrderedStatisticsCFAR(CFARDetector):
    """Ordered Statistics CFAR (OS-CFAR) detector."""
    
    def __init__(self, guard_cells: int, training_cells: int, pfa: float, k: int):
        """
        Initialize OS-CFAR detector.
        
        Args:
            guard_cells: Number of guard cells on each side
            training_cells: Number of training cells on each side
            pfa: Probability of false alarm
            k: Order statistic to use (k-th largest value)
        """
        super().__init__(guard_cells, training_cells, pfa)
        self.k = k
        
        # Calculate threshold factor
        self.threshold_factor = self._calculate_threshold_factor()
    
    def _calculate_threshold_factor(self) -> float:
        """Calculate threshold multiplier for OS-CFAR."""
        # Approximation for OS-CFAR threshold
        return -np.log(self.pfa)
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """Apply OS-CFAR detection."""
        detections = np.zeros_like(data, dtype=bool)
        
        # Pad data
        pad_width = self.guard_cells + self.training_cells
        padded_data = np.pad(data, pad_width, mode='edge')
        
        for i in range(len(data)):
            cut_idx = i + pad_width
            cut_value = padded_data[cut_idx]
            
            # Get training cells
            left_start = cut_idx - pad_width
            left_end = cut_idx - self.guard_cells
            right_start = cut_idx + self.guard_cells + 1
            right_end = cut_idx + pad_width + 1
            
            training_cells = np.concatenate([
                padded_data[left_start:left_end],
                padded_data[right_start:right_end]
            ])
            
            # Sort and select k-th value
            sorted_cells = np.sort(training_cells)
            noise_estimate = sorted_cells[-self.k]
            
            # Detection
            threshold = self.threshold_factor * noise_estimate
            detections[i] = cut_value > threshold
        
        return detections


def cfar_2d(data: np.ndarray, guard_cells: int = 2, training_cells: int = 8,
            pfa: float = 1e-6, method: str = 'ca') -> np.ndarray:
    """
    Apply 2D CFAR detection.
    
    Args:
        data: 2D input data (magnitude squared)
        guard_cells: Number of guard cells in each direction
        training_cells: Number of training cells in each direction
        pfa: Probability of false alarm
        method: CFAR method ('ca' for cell-averaging, 'os' for ordered statistics)
        
    Returns:
        Binary detection map
    """
    detections = np.zeros_like(data, dtype=bool)
    
    # Calculate threshold factor for CA-CFAR
    n_training = 4 * training_cells * (training_cells + 2 * guard_cells + 1) - \
                 4 * guard_cells * (guard_cells + 1)
    threshold_factor = n_training * (pfa**(-1/n_training) - 1)
    
    # Pad data
    pad_width = guard_cells + training_cells
    padded_data = np.pad(data, pad_width, mode='edge')
    
    # Apply 2D CFAR
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Cell under test indices in padded array
            cut_i = i + pad_width
            cut_j = j + pad_width
            cut_value = padded_data[cut_i, cut_j]
            
            # Define training region
            row_start = cut_i - pad_width
            row_end = cut_i + pad_width + 1
            col_start = cut_j - pad_width
            col_end = cut_j + pad_width + 1
            
            # Extract training window
            window = padded_data[row_start:row_end, col_start:col_end]
            
            # Create mask for training cells (exclude guard and CUT)
            mask = np.ones_like(window, dtype=bool)
            guard_start = pad_width - guard_cells
            guard_end = pad_width + guard_cells + 1
            mask[guard_start:guard_end, guard_start:guard_end] = False
            
            # Get training cells
            training_cells = window[mask]
            
            if method == 'ca':
                # Cell averaging
                noise_estimate = np.mean(training_cells)
            elif method == 'os':
                # Ordered statistics (use median)
                noise_estimate = np.median(training_cells)
            else:
                raise ValueError(f"Unknown CFAR method: {method}")
            
            # Detection decision
            threshold = threshold_factor * noise_estimate
            detections[i, j] = cut_value > threshold
    
    return detections


def adaptive_cfar(data: np.ndarray, guard_cells: int = 2, 
                  training_cells: int = 8, pfa: float = 1e-6) -> np.ndarray:
    """
    Adaptive CFAR that switches between CA and OS based on environment.
    
    Args:
        data: Input data
        guard_cells: Number of guard cells
        training_cells: Number of training cells
        pfa: Probability of false alarm
        
    Returns:
        Binary detection map
    """
    # Calculate threshold factors
    n_training = 2 * training_cells
    ca_threshold_factor = n_training * (pfa**(-1/n_training) - 1)
    os_k = max(1, int(0.75 * n_training))  # Use 75th percentile
    os_threshold_factor = n_training * (pfa**(-1/n_training) - 1) * 0.8  # Slightly lower for OS
    
    detections = np.zeros_like(data, dtype=bool)
    
    pad_width = guard_cells + training_cells
    padded_data = np.pad(data, pad_width, mode='edge')
    
    for i in range(len(data)):
        cut_idx = i + pad_width
        cut_value = padded_data[cut_idx]
        
        # Get training cells
        left_start = cut_idx - pad_width
        left_end = cut_idx - guard_cells
        right_start = cut_idx + guard_cells + 1
        right_end = cut_idx + pad_width + 1
        
        if left_start >= 0 and right_end <= len(padded_data):
            left_cells = padded_data[left_start:left_end]
            right_cells = padded_data[right_start:right_end]
            training_cells = np.concatenate([left_cells, right_cells])
            
            # Check heterogeneity
            mean_val = np.mean(training_cells)
            std_val = np.std(training_cells)
            cv = std_val / (mean_val + 1e-10)  # Coefficient of variation
            
            # Use OS-CFAR if heterogeneous (high CV)
            if cv > 0.5:
                # Ordered statistics
                sorted_cells = np.sort(training_cells)
                k_idx = min(os_k - 1, len(sorted_cells) - 1)
                noise_estimate = sorted_cells[k_idx]
                threshold = os_threshold_factor * noise_estimate
            else:
                # Cell averaging
                noise_estimate = mean_val
                threshold = ca_threshold_factor * noise_estimate
            
            detections[i] = cut_value > threshold
    
    return detections
