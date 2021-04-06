# -*- coding: utf-8 -*-

"""
Date: 23.06.2020
Author: Franziska Pleissner

          input/output  solar   thermal_h  thermal_l  electricity   water


collector source |------->|        |            |          |        |
                 |                 |            |          |        |
collector transf |<-------|        |            |          |        |
                 |<----------------------------------------|        |
                 |---------------->|            |          |        |
                 |                 |            |          |        |
power_block      |<----------------|            |          |        |
                 |----------------------------->|          |        |
                 |---------------------------------------->|        |
                 |        |        |            |          |        |
desalination     |<-----------------------------|          |        |
                 |<----------------------------------------|        |
                 |------------------------------------------------->|
                 |        |        |            |          |        |
aux_transformer  |<----------------|            |          |        |
                 |----------------------------->|          |        |
                 |        |        |            |          |        |
storage_water    |<-------------------------------------------------|
                 |------------------------------------------------->|
                 |        |        |            |          |        |
storage_heat     |<----------------|            |          |        |
                 |---------------->|            |          |        |
                 |        |        |            |          |        |
storage_         |<----------------------------------------|        |
electricity      |---------------------------------------->|        |
                 |        |        |            |          |        |
demand           |<-------------------------------------------------|
                 |        |        |            |          |        |
output_          |<----------------------------------------|        |
electricity      |        |        |            |          |        |
excess_heat      |<-----------------------------|          |        |
                 |        |        |            |          |        |

"""

############
# Preamble #
############

# Import packages
from oemof import solph
import oemof.solph.processing as processing
from oemof.tools import logger, economics
from oemof.thermal.concentrating_solar_power import csp_precalc

import logging
import os
import yaml
import pandas as pd
from datetime import datetime


def run_model_med(config_path, var_number):

    # Define some needed parameters
    currentdate = datetime.today().strftime('%Y%m%d')

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

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
    # data = pd.read_csv(data_ts_path + cfg['time_series_file_name'])

    # Change data for collector
    # date_rng = pd.date_range(start='1/1/2019', periods=number_of_time_steps,
    #                          freq='H', tz='Europe/Madrid')
    # col_data_2 = pd.read_csv(data_ts_path+cfg['time_series_file_name']).head(
    #     35040)
    # col_data = col_data_2.iloc[::4, :]
    #
    # col_data['Date'] = date_rng
    # col_data.set_index('Date', inplace=True)
    # col_data.to_csv(data_ts_path+'test_2.csv')
    # col_data = pd.read_csv(data_ts_path + 'test_2.csv')
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
            cfg['exp_number'], var_number, currentdate),
        header=False)

    # define costs function
    def ep_costs_f(capex, n, opex):
        ep_costs = (economics.annuity(capex, n, param_value['wacc'])
                    + capex * opex)
        return ep_costs
    ##################################

    # Initialise the energysystem
    date_time_index = collector_precalc_data.index
    energysystem = solph.EnergySystem(timeindex=date_time_index)

    # Initiate the logger
    logger.define_logging(
        logfile='desalination_med_{0}_{1}_{2}.log'.format(
            cfg['exp_number'], var_number, currentdate),
        logpath=results_path + '/logs',
        screen_level=logging.INFO,
        file_level=logging.DEBUG)

    #######################
    # Build up the system #
    #######################

    # busses
    bsol = solph.Bus(label='solar_bus')
    bthh = solph.Bus(label='thermal_high_bus')
    bthl = solph.Bus(label='thermal_low_bus')
    bele = solph.Bus(label='electricity_bus')
    bwat = solph.Bus(label='water_bus')

    energysystem.add(bsol, bthh, bthl, bele, bwat)

    # sinks and sources
    collector_source = solph.Source(
        label='collector_source',
        outputs={bsol: solph.Flow(
            max=collector_precalc_data['collector_heat'],
            investment=solph.Investment(
                ep_costs=ep_costs_f(
                    param_value['collector_invest_costs_output_th'],
                    param_value['collector_lifetime'],
                    param_value['collector_opex'])))})

    demand = solph.Sink(
        label='water_demand',
        inputs={bwat: solph.Flow(
            fix=dataframe['demand_water_in_qm'],
            nominal_value=1)})

    excess = solph.Sink(
        label='thermal_excess',
        inputs={bthl: solph.Flow()})

    electr_output = solph.Sink(
        label='electric_output',
        inputs={bele: solph.Flow(
            nominal_value=116708428800000,
            summed_max=1,
            variable_costs=param_value['selling_price_electricity'])})

    energysystem.add(collector_source, demand, excess, electr_output)

    # transformers
    collector_transformer = solph.Transformer(
        label='collector_transformer',
        inputs={
            bsol: solph.Flow(),
            bele: solph.Flow()},
        outputs={bthh: solph.Flow()},
        conversion_factors={
            bsol: 1,
            bele: param_value['collector_elec_consumption'],
            bthh: 1 - param_value['collector_thermal_losses']})

    powerplant = solph.Transformer(
        label='powerplant',
        inputs={bthh: solph.Flow()},
        outputs={
            bele: solph.Flow(
                investment=solph.Investment(
                    ep_costs=ep_costs_f(
                        param_value['powerplant_invest_costs'],
                        param_value['powerplant_lifetime'],
                        param_value['powerplant_opex']))),
            bthl: solph.Flow()},
        conversion_factors={
            bele: param_value['powerplant_conv_factor_electrical'],
            bthl: param_value['powerplant_conv_factor_thermal_out']})

    desalination = solph.Transformer(
        label='desalination',
        inputs={
            bthl: solph.Flow(),
            bele: solph.Flow()},
        outputs={bwat: solph.Flow(
            min=param_value['min_desalination'],
            investment=solph.Investment(
                ep_costs=ep_costs_f(
                    param_value['desalination_med_invest_costs'],
                    param_value['desalination_med_lifetime'],
                    param_value['desalination_med_opex']))
        )},
        conversion_factors={
            bthl: param_value['desalination_med_conv_factor_thermal'],
            bele: param_value['desalination_med_conv_factor_electric']})

    aux_transformer = solph.Transformer(
        label='auxiliary_transfomer',
        inputs={bthh: solph.Flow()},
        outputs={bthl: solph.Flow()},
        conversion_factors={bthl: 1})

    energysystem.add(collector_transformer, powerplant, desalination,
                     aux_transformer)

    # storages
    storage_thermal = solph.components.GenericStorage(
        label='storage_thermal',
        inputs={bthh: solph.Flow()},
        outputs={bthh: solph.Flow()},
        loss_rate=param_value['thermal_stor_losses'],
        inflow_conversion_factor=param_value['thermal_stor_conv_factor_in'],
        outflow_conversion_factor=param_value['thermal_stor_conv_factor_out'],
        investment=solph.Investment(
            ep_costs=ep_costs_f(
                param_value['thermal_stor_invest_costs_capacity'],
                param_value['thermal_stor_lifetime'],
                param_value['thermal_stor_opex'])))

    storage_water = solph.components.GenericStorage(
        label='storage_water',
        inputs={bwat: solph.Flow()},
        outputs={bwat: solph.Flow()},
        loss_rate=param_value['water_stor_losses'],
        inflow_conversion_factor=1,
        outflow_conversion_factor=1,
        investment=solph.Investment(
            ep_costs=ep_costs_f(
                param_value['water_stor_invest_costs'],
                param_value['water_stor_lifetime'],
                param_value['water_stor_opex'])))

    storage_electricity = solph.components.GenericStorage(
        label='storage_electricity',
        inputs={bele: solph.Flow()},
        outputs={bele: solph.Flow()},
        loss_rate=param_value['electric_stor_losses'],
        inflow_conversion_factor=param_value['electric_stor_conv_factor_in'],
        outflow_conversion_factor=param_value['electric_stor_conv_factor_out'],
        investment=solph.Investment(
            ep_costs=ep_costs_f(
                param_value['electric_stor_invest_costs_capacity'],
                param_value['electric_stor_lifetime'],
                param_value['electric_stor_opex'])))

    energysystem.add(storage_thermal, storage_water, storage_electricity)

    ########################################
    # Create a model and solve the problem #
    ########################################

    # Initialise the operational model (create the problem) with constrains
    model = solph.Model(energysystem)

    # Solve the model
    logging.info('Solve the optimization problem')
    model.solve(solver=solver, solve_kwargs={'tee': solver_verbose})

    if debug:
        filename = (results_path + '/lp_files/' +
                    'desalination_med_{0}_{1}_{2}.lp'.format(
                        cfg['exp_number'], var_number, currentdate))
        logging.info('Store lp-file in {0}.'.format(filename))
        model.write(filename, io_options={'symbolic_solver_labels': True})

    # save the results
    logging.info('Store the energy system with the results.')

    energysystem.results['main'] = solph.results(model)
    energysystem.results['meta'] = processing.meta_results(model)
    energysystem.results['param'] = (
        processing.parameter_as_dict(model))

    energysystem.dump(
        dpath=(results_path + '/dumps/'  + '/'),
        filename='desalination_med_{0}_{1}_{2}.oemof'.format(
            cfg['exp_number'], var_number, currentdate))
