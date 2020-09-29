# Desalination model

## Structur of the input data
There are two typs of input data: time series for the demand and the solar input
data and parameters to describe the components. Through historical reasons the
time series has to be provided in a folder called 'data_confidential' and the
parameters in a file called 'data_public'.

### Time series
The default name of the file is 'time_series.csv'. You can change the name, if you
change it also in the confoguration file. Time series to provide are:
* demand_water_in_qm
* Ambient_temperature_in_degC
* dni_in_W
* E_dir_hor_in_W
* t_amb
* global_horinzontal_in_W_m2

### Parameters
The parameters given in file 'parameters_experiment_desalination_base.csv' are
necessary. You can rename the file, but you have to change it in the configuration
file too.

## Structur of the configuration file
There is one configuration file called 'experiment_0'. You can add more and
execute them all at once by adding them to the 'main.py' file.

The file is a yaml-file, which specifies your experiment. 
It holds:
* experiments name and number
* the number of variation: it has to correspond with the length of
'parameters_system' list.

* Five run arguments, which decide, which file is executed and which not
* Two arguments, which decide, which method you want to use for the calculation
of the collector data. One has to be True, one has to be False. See the
documentation of oemof-thermal for the difference.

* Five parameters, which describe, how to do the optimisation.

* The file names for the input data.

You can provide 'parameters_system' as a list. For every item in the list,
there is calculated a new optimisation. The name of the results will be
<some_name>_experiment_number_variation_number.


