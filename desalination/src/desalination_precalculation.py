# -*- coding: utf-8 -*-

"""
Date: 23.06.2020
Author: Franziska Pleissner


"""

############
# Preamble #
############

# Import packages
from oemof.thermal.concentrating_solar_power import csp_precalc

import os
import yaml
import pandas as pd
from datetime import datetime


def run_precalculation(config_path, var_number):

    # Define some needed parameters
    currentdate = datetime.today().strftime('%Y%m%d')

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
#        cfg = yaml.load(ymlfile)

    solver = cfg['solver']
    debug = cfg['debug']
    solver_verbose = cfg['solver_verbose']  # show/hide solver output

    if debug:
        number_of_time_steps = 3
    else:
        number_of_time_steps = cfg['number_timesteps']

    # Data imports and preprocessing #
    ##################################
    # Define the used directories
    abs_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
    results_path = abs_path + '/results'
    data_ts_path = abs_path + '/data/data_timeseries/'
    data_param_path = abs_path + '/data/data_public/'

    # Read parameter values from parameter file
    file_path_param = data_param_path + cfg['parameters_system'][var_number]
    param_df = pd.read_csv(file_path_param, index_col=1)
    param_value = param_df['value']

    # Import weather and demand data
    dataframe = pd.read_csv(
        data_ts_path + cfg['time_series_file_name']).head(number_of_time_steps)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe.set_index('Date', inplace=True)
    dataframe = dataframe.asfreq(cfg['frequenz'])
    if cfg['time_series_file_name'] == 'oman.csv':
        dataframe.index = dataframe.index.tz_localize(tz='Asia/Muscat')

    # Calculate collector data
    if cfg['csp_method_normal']:
        collector_precalc_data = csp_precalc(
            param_value['latitude'],
            param_value['longitude'],
            param_value['collector_tilt'],
            param_value['collector_azimuth'],
            param_value['cleanliness'],
            param_value['eta_0'],
            param_value['c_1'],
            param_value['c_2'],
            param_value['temp_collector_inlet'],
            param_value['temp_collector_outlet'],
            dataframe['Ambient_temperature_in_degC'],
            param_value['a_1'],
            param_value['a_2'],
            param_value['a_3'],
            param_value['a_4'],
            param_value['a_5'],
            param_value['a_6'],
            loss_method='Andasol',
            irradiance_method='normal',
            dni=dataframe['dni_in_W'])

    if cfg['csp_method_horizontal']:
        collector_precalc_data = csp_precalc(
            param_value['latitude'],
            param_value['longitude'],
            param_value['collector_tilt'],
            param_value['collector_azimuth'],
            param_value['cleanliness'],
            param_value['eta_0'],
            param_value['c_1'],
            param_value['c_2'],
            param_value['temp_collector_inlet'],
            param_value['temp_collector_outlet'],
            dataframe['Ambient_temperature_in_degC'],
            param_value['a_1'],
            param_value['a_2'],
            param_value['a_3'],
            param_value['a_4'],
            param_value['a_5'],
            param_value['a_6'],
            loss_method='Andasol',
            E_dir_hor=dataframe['E_dir_hor_in_W'])

    collector_precalc_data.to_csv(
        results_path + '/precalcs/precalc' + '_{0}_{1}_{2}.csv'.format(
            cfg['exp_number'], var_number, currentdate))
