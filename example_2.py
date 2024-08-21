import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matching_pursuit import (
    matching_pursuit, generate_combined_random_signal,
    DICTIONARY
)

# v v v v SETUP v v v v
# Setup the windows 2 plot figures
window, (figure1, figure2) = plt.subplots(1, 2, figsize=(14, 5))

# this function will run on each iteration of the animation
def animation_plot(iteration_number):

    # Clear previous plots
    figure1.clear()
    figure2.clear()
    
    # Use the output_signal from matching_pursuit to ensure consistency
    current_applied_changes = np.zeros_like(input_signal)
    
    # Accumulate the reconstructed signal only up to the current iteration `i`
    for j in range(iteration_number):
        current_applied_changes += coefficients[j] * DICTIONARY[indices[j]]
    
    current_signal = input_signal + current_applied_changes  # Accumulate the reconstructed signal
    
    figure1.plot(current_signal, color='orange', linestyle='dashed', label='Reconstructed Signal')
    figure1.plot(input_signal, color='grey', label='Input Signal')
    figure1.plot(expected_signal, color='blue', linestyle='dotted', label='Expected Signal')
    figure1.set_xlim([0, len(input_signal)])
    figure1.set_ylim([min(expected_signal)-1, max(expected_signal)+1])
    figure1.set_title("Reconstructed Signal after {} Iterations".format(iteration_number))
    figure1.legend()
    

    figure2.stem(
        indices[:iteration_number+1], # [X-AXIS]
        coefficients[:iteration_number+1], # [Y-AXIS]
        linefmt='purple', markerfmt='x', basefmt=' '
    )
    figure2.set_xlim([0, len(DICTIONARY)])
    figure2.set_title("Coefficients after {} Iterations".format(iteration_number))
    figure2.set_xlabel("Element of the basis")
    figure2.set_ylabel("Coefficient")
# ^ ^ ^ ^ SETUP ^ ^ ^ ^


# v v v v IMPLEMNTATION v v v v
input_signal = np.linspace(0, 0, 100) # simple straight line
expected_signal = generate_combined_random_signal()

# Run the matching pursuit algorithm
_, coefficients, indices = matching_pursuit(
    input_signal, expected_signal,
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
