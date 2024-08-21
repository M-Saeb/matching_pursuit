# Matching Pursuit
## Introduction
This code is a an implementatino of the Matching Pursuit Alogorithim using Python 

## Run the code
If you wish to run the code local on your machine you need :

- **Languges:** Python3
- **Libraries** `matplotlib` and `numpy` which you can install by running the command 
    ```bash
    python -m pip install -r pip_requirements.txt
    ```

if the installation was successfull you should be able to run the examples in the project using the commands
```bash
python example_1.py
# or
python example_2.py
```

## Layout
The code base is structed like the following:

### 1. Matching Pursuit Folder
In this folder you'll find the main function for running the matching pursuit algorithim

### 2. example_1.py
The code for running a plot for the matching pursuit algorithim which shows 4 signals
1. The input signal
2. The expected output signal
3. The required change to for input signal to transform to the expected output
4. the actual output signal was the combiniation of signals 1 and 3

### 3. example_2.py
An animation showing how output signal is updated with each iteration and the coeffient needed for it