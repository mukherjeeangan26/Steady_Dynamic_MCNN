The excel spreadsheets provided here include steady-state and dynamic datasets for developing the MCNN model of the 
isothermal continuous stirred tank reactor (CSTR) system using different types of noise characterizations in 
training data.

## Steady-State CSTR Data

This spreadsheet contains 400 steady-state input-output datasets for the CSTR system. The model inputs are represented 
by the inlet feed space velocity (F/V), and the concentration of all four reaction species in the feed stream (CAf, CBf, 
CCf, CDf), whereas, the model outputs are denoted by the outlet concentration of all reaction species in the product 
stream, i.e., CA, CB, CC, CD.

The .xlsx file contains 4 tabs in total. The last tab contains a schematic of the CSTR system under consideration. 
The first three tabs contain steady-state data for the different types of noise / uncertainties added to the simulated
data to generate training data, i.e., no noise (where truth and measurements are the same), constant bias with an 
additional zero-mean Gaussian noise, and random bias with an additional Gaussian noise distribution, respectively.


## Dynamic CSTR Data

This spreadsheet contains dynamic time-series (total duration of around 8000 time steps) input-output datasets for 
the CSTR system. The model inputs are represented by the inlet feed space velocity (F/V), and the concentration of 
all four reaction species in the feed stream (CAf, CBf, CCf, CDf), whereas, the model outputs are denoted by the 
outlet concentration of all reaction species in the product stream, i.e., CA, CB, CC, CD.

The .xlsx file contains 5 tabs in total. The last tab contains a schematic of the CSTR system under consideration. 
The first three tabs contain dynamic data for the different types of noise / uncertainties added to the simulated
data to generate training data, i.e., no noise (where truth and measurements are the same), time-invariant bias with 
an additional zero-mean Gaussian noise, and time-varying bias with an additional Gaussian noise distribution, respectively.
The fourth tab contains the data required for developing dynamic MCNNs under the assumption that system holdup information
is available. In this case, the training data for model outputs contain a time-invariant bias in presence of an additional
Gaussian noise distribution. The optimal MCNN developed in this specific case of the CSTR under consideration only 
considers a single input, i.e., instantaneous volume of the reactor (V) which, along with the constant (time-invariant) 
volumetric flow rate and inlet feed concentrations, account for the system holdup information to be utilized during model 
development. The model output variables remain the same as the previous cases.


The input and output variables have been color-coded in both excel spreadsheets attached for clarity --
  * The columns representing the variation in model inputs shown in 'light orange'
  * The columns representing the variation in corresponding model outputs shown in 'light green'
  * The columns representing the constant (time-invariant) model inputs (e.g., for the specific holdup case
    considered here) shown in 'light cyan'.
