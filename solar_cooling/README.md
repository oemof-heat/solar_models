# Solar cooling model

## Structur of the input data
There are two typs of input data: time series for the demand and the solar input
data and parameters to describe the components. Through historical reasons the
time series has to be provided in a folder called 'data_confidential' and the
parameters in a file called 'data_public'.

Time series to provide are:
* Cooling load kW
* global_irradiance_kW_per_m2_TMY
* diffus_irradiance_kW_per_m2_TMY
* pv_normiert
* t_amb

('pv_normiert' is the output in kW of a pv-modul with 0.9708 kWpeak)

There are two files for the parameters: One file holds the parameters and another
file holds the factors to vary some parameters. This way, it is easy to investigate
the influence of this parameters in an experiment. The files are named:
* parameters_experiment_0.csv
* parameters_variation_base.csv