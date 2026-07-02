This directory contains the EPED neural network used for high-field FPP scoping studies, trained by Aaron Ho (MIT)

This is an ensemble of 10 feed-forward NNs, each with 11 inputs, 2 outputs, and 3 hidden layers of [200, 100, 50] neurons (GELU activation)

The minimum and maximum of the training dataset input ranges used to generate these models are below, in order of input position:

a:      [  0.4  ,   2.2  ]
aspect: [  2.0  ,   4.2  ]
kappa:  [  1.3  ,   2.5  ]
delta:  [  0.3  ,   0.7  ]
bt:     [  2.0  ,  18.0  ]  * Not a clean boundary so (3.0, 17.0) might be more prudent
qstar:  [  3.0  ,   5.0  ]
betan:  [  0.3  ,   3.7  ]
zeff:   [  1.2  ,   3.2  ]
fgped:  [  0.3  ,   1.3  ]
nsfrac: [  0.2  ,   0.8  ]
tesep:  [ 50.0  , 500.0  ]

Both kappa and delta were taken from the normalized poloidal flux surface (psi) = 0.995 for the training dataset

A simple test of the NN is provided in "test_script.py" inside this directory, from which an example code for loading and evaluating the network can also be taken

A citation for this model is pending
