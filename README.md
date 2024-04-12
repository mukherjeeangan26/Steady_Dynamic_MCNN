# Steady-State and Dynamic Mass Constrained Neural Networks (MCNN)
MATLAB Codes and Python Notebooks for developing steady-state and dynamic Mass Constrained Neural Network (MCNN) models using noisy transient data

Author: Angan Mukherjee (am0339@mix.wvu.edu)

Last Page Update: April 12, 2024

# Announcement

We are very welcome to your contribution. Please feel free to reach out to us with your feedback and suggestions on how to improve the current models.

# Publication

This public repository contains MATLAB codes and Python notebooks for developing and simulating different steady-state and dynamic MCNNs for nonlinear 
chemical systems using noisy transient data. The corresponding publication for this work is:

**Mukherjee, A.** & Bhattacharyya, D. "*On the Development of Steady-State and Dynamic Mass-Constrained Neural Networks Using Noisy Transient Data*". 
Comput. Chem. Eng. (Under Review)

These codes will be updated in subsequent versions to enhance robustness of the proposed algorithms and user friendliness.

# Brief Description

## Sample Data

Test case steady-state and dynamic data have been provided in Excel spreadsheets with respect to the operation of a continuous stirred tank reactor (CSTR)
system. Note that the codes uploaded in this repository are generic and can be applied to create mass-constrained models for any steady-state and dynamic
systems (data). The training and validation / simulation datasets need to be loaded at the beginning while running the inverse problem / forward problem
codes, indexed as '_MainFile.m'. The rows of the input and output data matrices refer to the observation indices and time steps for steady-state and 
dynamic modeling respectively while the corresponding columns signify the different input / output variables.

## Pre-Requisites for MATLAB and Python Codes

### Pre-Requisites for MATLAB Codes

For the models developed in this work, MATLAB's default optimization solver for constrained optimization problems (i.e., '*fmincon*') seldom led to convergence
issues, especially during dynamic optimization. Therefore, it is desired to implement Interior Point with Filter line search algorithm (IPOPT) for estimation of
optimal parameters for the equality constrained network models. Therefore, the IPOPT solver has been implemented through the OPTimization Interface (OPTI) Toolbox 
for both steady-state and dynamic MCNN. The open-sourced OPTI Toolbox can be accessed at the following link: https://github.com/jonathancurrie/OPTI

The user requries to install OPTI in the MATLAB window by running the **opti_Install.m** and following recommended specifications. 

### Pre-Requisites for Python Codes

The equality constrained optimization problems have been solved using the Python-based open-source optimization modeling language, Pyomo v6.7.1 through the IDAES framework.

More details about Pyomo can be found at: https://github.com/Pyomo/pyomo

More details about IDAES can be found at: https://github.com/IDAES/idaes-pse

One may follow the following installation instructions for downloading Python / Pyomo / IDAES:
  * Install Anaconda: https://www.anaconda.com/download
  * Run 'Anaconda Prompt'
  * Create a new environment by: conda create -n my-new-env
  * Activate the new environment by: conda activate my-new-env
  * Install IDAES by: pip install idaes-pse
  * Get IDAES extensions by: idaes get-extensions

Other libraries may be installed before running the Python notebooks, as necessary.

In addition, to access the training / validation codes for the unconstrained all-nonlinear static-dynamic neural networks used in the dynamic MCNNs, please refer to the 
following link: https://github.com/mukherjeeangan26/Hybrid_StaticDynamic_NN_Models. 
Additional details about the unconstrained all-nonlinear network architectures and training algorithms have been discussed in the following publication:

**Mukherjee, A.** & Bhattacharyya, D. "*Hybrid Series/Parallel All-Nonlinear Dynamic-Static Neural Networks: Development, Training, and Application to Chemical Processes*". 
Ind. Eng. Chem. Res. 62, 3221–3237 (2023). DOI: https://pubs.acs.org/doi/full/10.1021/acs.iecr.2c03339






