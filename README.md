These files are input/exec/output files for machine-learning empirical pseudopotentials.

band: A collection of codes that calculate the potential values of the target system using the trained model.

data: A collection of Python codes that include the process of generating input and post-processing for DFT calculations (QE) to create training data.

ml: Python code for training an artificial neural network model. It is divided into two versions: one for cases where the G cutoff is large and another for when it is small.
