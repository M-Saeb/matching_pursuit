import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matching_pursuit import (
    matching_pursuit, generate_combined_random_signal,
    DICTIONARY
)

# v v v v SETUP v v v v
INPUT_SIGNAL = np.linspace(0, 0, 100) # simple straight line
EXPECTED_SIGNAL = generate_combined_random_signal()

# Setup the windows 2 plot figures
window, (figure1, figure2) = plt.subplots(1, 2, figsize=(14, 5))

# this function will run on each iteration of the animation
def animation_plot(frame_number):

    # Clear previous plots
    figure1.clear()
    figure2.clear()
    
    # Use the output_signal from matching_pursuit to ensure consistency
    current_applied_changes = np.zeros_like(INPUT_SIGNAL)
    
    # Accumulate the reconstructed signal only up to the current iteration `i`
    for j in range(frame_number):
        current_applied_changes += coefficients[j] * DICTIONARY[indices[j]]
    
    current_signal = INPUT_SIGNAL + current_applied_changes  # Accumulate the reconstructed signal
    
    figure1.plot(INPUT_SIGNAL, color='grey', label='Input Signal')
    figure1.plot(current_signal, color='orange', linestyle='dashed', label='Reconstructed Signal')
    figure1.plot(EXPECTED_SIGNAL, color='blue', linestyle='dotted', label='Expected Signal')
    figure1.set_xlim([0, len(INPUT_SIGNAL)])
    figure1.set_ylim([min(EXPECTED_SIGNAL)-1, max(EXPECTED_SIGNAL)+1])
    figure1.set_title("Reconstructed Signal after {} Iterations".format(frame_number))
    figure1.legend()
    
    if frame_number > 0: # This condition is to not plot any coefficient on the first frame
        figure2.stem(
            indices[:frame_number], # [X-AXIS]
            coefficients[:frame_number], # [Y-AXIS]
            linefmt='purple', markerfmt='x', basefmt=' '
        )
    figure2.set_xlim([0, len(DICTIONARY)])
    figure2.set_title("Coefficients after {} Iterations".format(frame_number))
    figure2.set_xlabel("Element of the basis")
    figure2.set_ylabel("Coefficient")
# ^ ^ ^ ^ SETUP ^ ^ ^ ^


# v v v v IMPLEMNTATION v v v v
# Run the matching pursuit algorithm
_, coefficients, indices = matching_pursuit(
    INPUT_SIGNAL, EXPECTED_SIGNAL,
    max_iter=101
)

# Create animation
animate = FuncAnimation(
    window,
    animation_plot,
    frames=len(coefficients),
    interval=500, repeat=False
)
plt.show() # keep the plot open
# ^ ^ ^ ^ IMPLEMNTATION ^ ^ ^ ^
