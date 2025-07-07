import typing as tp

import numpy as np

from .constants import STRING, INTEGER, PLAIN, NEIGHBOR, RELATIVE, ABSOLUTE

ALLOWED_FEATURES = [STRING, INTEGER]
ALLOWED_EMBEDDINGS = [PLAIN, NEIGHBOR]
ALLOWED_POSITION_EMBEDDINGS = [RELATIVE, ABSOLUTE]

def check_feature_type(feature_type: str) -> None:
    if feature_type not in ALLOWED_FEATURES:
            raise ValueError(f"`feature_type` must be in {ALLOWED_FEATURES}, got {feature_type}")

def check_embedding_type(embedding_type: str) -> None:
    if embedding_type not in ALLOWED_EMBEDDINGS:
            raise ValueError(f"`feature_type` must be either {ALLOWED_EMBEDDINGS}, got {embedding_type}")

def check_position_embedding_type(position_embedding_type: str) -> None:
    if position_embedding_type not in ALLOWED_POSITION_EMBEDDINGS:
            raise ValueError(f"`feature_type` must be either {ALLOWED_POSITION_EMBEDDINGS}, got {position_embedding_type}")

class HiPPOMatrix:
    """Generate HiPPO matrices for different polynomial bases."""
    
    @staticmethod
    def legendre(N: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Generate HiPPO-LegT matrix (Legendre polynomials, translated).

        Args:
            N: Dimension of the state space
            
        Returns:
            Tuple of (A, B) matrices for HiPPO-LegT
        """
        A = np.zeros((N, N))
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = np.sqrt((2*n+1)*(2*k+1))
                elif n == k:
                    A[n, k] = 2*n + 1
                else:
                    A[n, k] = 0
        return (-A, A)
    
    @staticmethod
    def legendre_scaled(N: int, theta: float = 1.0) ->  tp.Tuple[np.ndarray, np.ndarray]:
        """
        Generate HiPPO-LegS matrix (Legendre Scaled) for sliding window.
        
        Args:
            N: Dimension of the state space
            theta: Scaling parameter (window size)
            
        Returns:
            Tuple of (A, B) matrices for HiPPO-LegS
        """
        A = np.zeros((N, N))
        
        # HiPPO-LegS A matrix
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = np.sqrt((2*n+1)*(2*k+1))
                elif n == k:
                    A[n, k] = n + 1
                else:
                    A[n, k] = 0
        
        return (-A / theta, A / theta)
    
    @staticmethod
    def laguerre(N: int) -> np.ndarray:
        """Generate HiPPO-LagT matrix (Laguerre polynomials, translated)."""
        A = np.zeros((N, N))
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = -1
                elif n == k:
                    A[n, k] = -(2*n + 1)
                else:
                    A[n, k] = 0
        return A
    
    @staticmethod
    def fourier(N: int) -> np.ndarray:
        """Generate HiPPO-FouT matrix (Fourier basis)."""
        A = np.zeros((N, N))
        for n in range(N):
            for k in range(N):
                if n == k:
                    A[n, k] = 0
                elif (n + k) % 2 == 1:  # n + k is odd
                    A[n, k] = 2 * np.pi * (n - k) / (n + k + 1)
        return A
