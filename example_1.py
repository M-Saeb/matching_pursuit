import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matching_pursuit import (
    matching_pursuit, generate_combined_random_signal
)

# v v v v SETUP v v v v
INPUT_SIGNAL = np.sin(np.linspace(0, 2 * np.pi, 100))
EXPECTED_SIGNAL =  generate_combined_random_signal()

# Creat the plot window
window = plt.figure(figsize=(10, 5))
plt.title('Matching Pursuit')

def animation_plot(frame_number):
    window.clear()
    plt.plot(INPUT_SIGNAL, label='Input Signal', color="orange")
    if frame_number >= 1:
        plt.plot(EXPECTED_SIGNAL, label='Expected Signal', color='black', linestyle='dashed')
    if frame_number >= 2:
        plt.plot(required_changes, label='Required Change', color='green', linestyle='dashdot')
    if frame_number >= 3:
        plt.plot(output_signal, label='Final Signal (Input + Required Change)', color='red', linestyle='dotted')
    plt.legend()
# ^ ^ ^ ^ SETUP ^ ^ ^ ^

# v v v v IMPLEMNTATION v v v v
# Run the matching pursuit algorithm
required_changes, _, _ = matching_pursuit(
    INPUT_SIGNAL, EXPECTED_SIGNAL
)

# Calculate the final signal by adding the required change to the input signal
output_signal = INPUT_SIGNAL + required_changes

# Create animation
animate = FuncAnimation(
    window,
    animation_plot,
    frames=4,
    interval=2000,
    repeat=False
)

plt.tight_layout()
plt.show()
# ^ ^ ^ ^ IMPLEMNTATION ^ ^ ^ ^
