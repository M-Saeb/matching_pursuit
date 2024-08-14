import matplotlib.pyplot as plt
import numpy as np

def matching_pursuit(input_signal: np.ndarray, expected_signal: np.ndarray, dictionary: np.ndarray, max_iter: int = 1000, tolerance: float = 1e-6):
    # Initialize the residual as the difference between expected signal and input signal
    residual = expected_signal - input_signal
    
    # Initialize an empty array for the output signal
    output_signal = np.zeros_like(input_signal)
    
    # Normalize the dictionary to avoid issues with large or small values
    dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    
    for _ in range(max_iter):
        # Calculate the correlation of the residual with each dictionary element
        correlations = np.dot(dictionary, residual)
        
        # Find the dictionary element with the highest correlation (best match)
        best_match_idx = np.argmax(np.abs(correlations))
        best_match = dictionary[best_match_idx]
        
        # Calculate the coefficient for this dictionary element
        coefficient = correlations[best_match_idx]
        
        # Update the output signal with this component
        output_signal += coefficient * best_match
        
        # Update the residual
        residual = expected_signal - input_signal - output_signal
        
        # Check if the residual is small enough to stop
        if np.linalg.norm(residual) < tolerance:
            break
    
    # Return the difference that needs to be added to input signal to match the expected signal
    return output_signal


# Input signal (a simple sine wave)
input_signal = np.sin(np.linspace(0, 2 * np.pi, 100))

# Expected signal
# expected_signal = np.sin(np.linspace(0, 2 * np.pi, 100)) + 0.2 * np.random.randn(100)
expected_signal =  np.cos(np.linspace(0, 4 * np.pi, 100)) 

# Simple dictionary of sine and cosine waves
dictionary = np.array([
    np.sin(np.linspace(0, 2 * np.pi, 100)),
    np.sin(np.linspace(0, 4 * np.pi, 100)),
    np.cos(np.linspace(0, 2 * np.pi, 100)),
    np.cos(np.linspace(0, 4 * np.pi, 100)),
])

# Run the matching pursuit algorithm
required_changes = matching_pursuit(input_signal, expected_signal, dictionary)

# Calculate the final signal by adding the required change to the input signal
final_signal = input_signal + required_changes

# Plotting the signals
plt.figure(figsize=(10, 5))

plt.plot(input_signal, label='Input Signal', color="orange")
plt.legend()
plt.title('Input Signal')

plt.plot(expected_signal, label='Expected Signal', color='black', linestyle='dashed')
plt.legend()
plt.title('Expected Signal')

plt.plot(required_changes, label='Required Change', color='green', linestyle='dashdot')
plt.legend()
plt.title('Required Change to Match Expected Signal')

plt.plot(final_signal, label='Final Signal (Input + Required Change)', color='red', linestyle='dotted')
plt.legend()
plt.title('Final Signal (Input + Required Change)')

# Display the plots
plt.tight_layout()
plt.show()