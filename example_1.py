import matplotlib.pyplot as plt
import numpy as np
from matching_pursuit import (
    matching_pursuit, generate_combined_random_signal,
    DICTIONARY
)


# Input signal (a simple sine wave)
input_signal = np.sin(np.linspace(0, 2 * np.pi, 100))

expected_signal =  generate_combined_random_signal()


# Run the matching pursuit algorithm
required_changes, _, _ = matching_pursuit(input_signal, expected_signal)

# Calculate the final signal by adding the required change to the input signal
final_signal = input_signal + required_changes

# Plotting the signals
plt.figure(figsize=(10, 5))

plt.plot(input_signal, label='Input Signal', color="orange")

plt.plot(expected_signal, label='Expected Signal', color='black', linestyle='dashed')

plt.plot(required_changes, label='Required Change', color='green', linestyle='dashdot')

plt.plot(final_signal, label='Final Signal (Input + Required Change)', color='red', linestyle='dotted')
plt.title('Matching Pursuit')

plt.tight_layout()
plt.legend()
plt.show()
