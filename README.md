# Implementation of Artificial Neural Network
Implementation of Artificial Neural Network with Bionodal activation functions in Python.

I use batch gradient descent with momentum in learning phase, momentum can be controlled by beta hyperparameter. For regularization I use L2 in every layer, controlled by reg_lambda hyperparameter. Parameter "r" controls non-linearity of activation funcions, for further details about this group of functions take a look into this repositary: https://github.com/marek-kan/Bionodal-root-units

 * test.py: - runs simple test on sci-kit learn boston housing dataset. This NN beats my Liner Regression model by ~1 MAE (~2.2 vs ~3.3).
 * bru_regressor.py - contains a model definition.
