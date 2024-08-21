import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def matching_pursuit(input_signal: np.ndarray, expected_signal: np.ndarray, dictionary: np.ndarray, max_iter: int = 1000, tolerance: float = 1e-6):
    # Initialize the residual as the difference between expected signal and input signal
    residual = expected_signal - input_signal
    
    # Initialize an empty array for the output signal
    output_signal = np.zeros_like(input_signal)
    
    indices = []
    coefficients = []

    for _ in range(max_iter):
        # Calculate the correlation of the residual with each dictionary element
        correlations = np.dot(dictionary, residual)
        
        # Find the dictionary element with the highest correlation (best match)
        best_match_idx = np.argmax(np.abs(correlations))
        best_match = dictionary[best_match_idx]
        
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
    
    # Return the output signal, residual, coefficients, and indices
    return output_signal, residual, coefficients, indices


# Example usage:

# Input signal (a simple sine wave)
input_signal = np.linspace(0, 0, 100)
# input_signal = np.sin(np.linspace(0, 2 * np.pi, 100))


# Simple dictionary of sine and cosine waves
dictionary = np.array([
    np.sin(np.linspace(0, 2 * np.pi, 100)),
    np.sin(np.linspace(0, 4 * np.pi, 100)),
    np.sin(np.linspace(0, 5 * np.pi, 100)),
    np.sin(np.linspace(0, 6 * np.pi, 100)),
    np.cos(np.linspace(0, 2 * np.pi, 100)),
    np.cos(np.linspace(0, 4 * np.pi, 100)),
    np.cos(np.linspace(0, 5 * np.pi, 100)),
    np.cos(np.linspace(0, 6 * np.pi, 100)),
])

# Expected signal (a sine wave with some added noise)
expected_signal = dictionary[random.randint(0, len(dictionary-1))] + \
                dictionary[random.randint(0, len(dictionary-1))] + \
                dictionary[random.randint(0, len(dictionary-1))] + \
                dictionary[random.randint(0, len(dictionary-1))]

# Normalize the dictionary to avoid issues with large or small values
dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)

# Run the matching pursuit algorithm
required_changes, _, coefficients, indices = matching_pursuit(
    input_signal, expected_signal, dictionary,
    max_iter=100
)

output_signal = input_signal + required_changes


def animate(i):
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Use the output_signal from matching_pursuit to ensure consistency
    current_applied_changes = np.zeros_like(input_signal)
    
    # Accumulate the reconstructed signal only up to the current iteration `i`
    for j in range(i):
        current_applied_changes += coefficients[j] * dictionary[indices[j]]
    
    current_signal = input_signal + current_applied_changes  # Accumulate the reconstructed signal
    
    ax1.plot(current_signal, color='orange', linestyle='dashed', label='Reconstructed Signal')
    ax1.plot(input_signal, color='red', label='Input Signal')
    ax1.plot(expected_signal, color='blue', linestyle='dotted', label='Expected Signal')
    ax1.set_xlim([0, len(input_signal)])
    ax1.set_ylim([min(expected_signal)-1, max(expected_signal)+1])
    ax1.set_title("Reconstructed Signal after {} Iterations".format(i))
    ax1.legend()
    

    # ax2.stem([X-AXIS], [Y-AXIS])
    # ax2.stem([1, 2], [1, 1], 'purple', markerfmt='o', basefmt=' ')
    ax2.stem(indices[:i+1], coefficients[:i+1], linefmt='purple', markerfmt='x', basefmt=' ')
    ax2.set_xlim([0, len(dictionary)])
    # ax2.set_ylim([-1, 1])
    ax2.set_title("Coefficients after {} Iterations".format(i))
    ax2.set_xlabel("Element of the basis")
    ax2.set_ylabel("Coefficient")


# Setup the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Create animation
ani = FuncAnimation(fig, animate, frames=len(coefficients), interval=500, repeat=False)

# Save or display the animation
plt.show()
