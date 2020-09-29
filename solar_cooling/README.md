# Solar cooling model

## Structur of the input data
There are two typs of input data: time series for the demand and the solar input
data and parameters to describe the components. Through historical reasons the
time series has to be provided in a folder called 'data_confidential' and the
parameters in a file called 'data_public'.

### Time series
The default name of the file is 'time_series.csv'. You can change the name, if you
change it also in the confoguration file. Time series to provide are:
* Cooling load kW
* global_irradiance_kW_per_m2_TMY
* diffus_irradiance_kW_per_m2_TMY
* t_amb
* pv_normiert (it is the output in kW of a pv-modul with 0.9708 kWpeak)

### Parameters
There are two files for the parameters: One file holds the parameters and another
file holds the factors to vary some parameters. This way, it is easy to investigate
the influence of this parameters in an experiment. The files are named:
* parameters_experiment_0.csv
* parameters_variation_base.csv

## Structur of the configuration file
There is one configuration file called 'experiment_0'. You can add more and
execute them all at once by adding them to the 'main.py' file.

The file is a yaml-file, which specifies your experiment. 
It holds:
* experiments name and number
* the number of variation: it has to correspond with the length of
'parameters_variation' list or the length of 'parameters_system' list.
.

* For run arguments, which decide, which file is executed and which not
* Four parameters, which describe, how to do the optimisastion.

* The file names for the input data.

You can provide 'parameters_system' or 'parameters_variation' as a list. The
other one must be no list. For every item in the list, there is calculated a
new optimisation. The name of the results will be
<some_name>_experiment_number_variation_number.


