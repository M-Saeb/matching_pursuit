import random
import numpy as np

_dictionary = np.array([  # Simple array of sine and cosine waves
    np.sin(np.linspace(0, 2 * np.pi, 100)),
    np.sin(np.linspace(0, 4 * np.pi, 100)),
    np.sin(np.linspace(0, 5 * np.pi, 100)),
    np.sin(np.linspace(0, 6 * np.pi, 100)),
    np.sin(np.linspace(0, 8 * np.pi, 100)),
    np.cos(np.linspace(0, 2 * np.pi, 100)),
    np.cos(np.linspace(0, 4 * np.pi, 100)),
    np.cos(np.linspace(0, 5 * np.pi, 100)),
    np.cos(np.linspace(0, 6 * np.pi, 100)),
    np.cos(np.linspace(0, 8 * np.pi, 100)),
])

# Normalizing the dictionary to avoid issues with large or small values
DICTIONARY = _dictionary / np.linalg.norm(_dictionary, axis=1, keepdims=True)

def generate_combined_random_signal() -> np.ndarray[np.float64]:
    # combining 4 signals from the dictionary to make one signal
    return DICTIONARY[random.randint(0, len(DICTIONARY)-1)] + \
        DICTIONARY[random.randint(0, len(DICTIONARY)-1)] + \
        DICTIONARY[random.randint(0, len(DICTIONARY)-1)] + \
        DICTIONARY[random.randint(0, len(DICTIONARY)-1)]


def matching_pursuit(
    input_signal: np.ndarray,
    expected_signal: np.ndarray,
    max_iter: int = 1000,
    tolerance: float = 1e-6
):
    # Initialize the residual as the difference between expected signal and input signal
    residual = expected_signal - input_signal
    
    # Initialize an empty array for the output signal
    output_signal = np.zeros_like(input_signal)
    
    indices = []
    coefficients = []

    for _ in range(max_iter):
        # Calculate the correlation of the residual with each dictionary element
        correlations = np.dot(DICTIONARY, residual)
        
        # Find the dictionary element with the highest correlation (best match)
        best_match_idx = np.argmax(np.abs(correlations))
        best_match = DICTIONARY[best_match_idx]
        
        # Calculate the coefficient for this dictionary element
        coefficient = correlations[best_match_idx]
        indices.append(best_match_idx)
        coefficients.append(coefficient)

        # Update the output signal with this component
        output_signal += coefficient * best_match
        
        # Update the residual
        residual = expected_signal - input_signal - output_signal
        
        # Check if the residual is small enough to stop
        if np.linalg.norm(residual) < tolerance:
            break
    
    return output_signal, coefficients, indices
