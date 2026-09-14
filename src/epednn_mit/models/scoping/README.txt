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

Derived quantities: several of the inputs are not raw simulation/equilibrium quantities but are derived from them. Using:
  a      = minor radius [m]
  r      = major radius [m]
  ip     = plasma current [MA]
  bt     = toroidal field [T]
  kappa  = elongation []
  delta  = triangularity []
  neped  = electron density at the pedestal [1e19 m^-3]
  nesep  = electron density at the separatrix [1e19 m^-3]

the derived inputs are computed as:

  aspect  = r / a
  epsilon = a / r
  shaping = 0.5 * (1.0 + kappa^2 * (1.0 + 2.0*delta^2 - 1.2*delta^3)) * (1.17 - 0.65*epsilon) / (1.0 - epsilon^2)^2
  fgped   = neped * pi * a^2 / (10.0 * ip)
  qstar   = 5.0 * a^2 * bt * shaping / (r * ip)
  nsfrac  = nesep / neped

fgped is the Greenwald fraction of neped (i.e. neped / n_Greenwald, with n_Greenwald = ip / (pi * a^2) in units of 1e20 m^-3). qstar is the shaped (engineering) edge safety factor of Uckan & Sauthoff (ITER Physics Design Guidelines, 1990), which depends on elongation, triangularity, and inverse aspect ratio through the "shaping" factor above -- it is not the simple cylindrical q* (2*pi*a^2*kappa*bt / (mu0*r*ip)), which omits the delta dependence and is no longer used.

A simple test of the NN is provided in "*_model.py" scripts inside this directory, from which an example code for loading and evaluating the network can also be taken

A citation for this model is pending
