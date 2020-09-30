# -*- coding: utf-8 -*-

"""
Date: 23.06.2020
Author: Franziska Pleissner

          input/output  electricity   water


pv               |--------->|          |
                 |                     |
desalination     |<---------|          |
                 |-------------------->|
                 |          |          |
storage_water    |<--------------------|
                 |-------------------->|
                 |          |          |
storage_         |<---------|          |
electricity      |--------->|          |
                 |          |          |
demand           |<--------------------|
                 |          |          |
output_          |<---------|          |
electricity      |          |          |

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


def run_model_ro(config_path, var_number):

    # Define some needed parameters
    currentdate = datetime.today().strftime('%Y%m%d')

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    solver = cfg['solver']
    debug = cfg['debug']
    solver_verbose = cfg['solver_verbose']  # show/hide solver output

    if debug:
        number_of_time_steps = 5
    else:
        number_of_time_steps = cfg['number_timesteps']

    # Data imports and preprocessing #
    ##################################
    # Define the used directories
    abs_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
    results_path = abs_path + '/results'
    data_ts_path = abs_path + '/data/data_confidential/'
    data_param_path = abs_path + '/data/data_public/'

    # Read parameter values from parameter file
    file_path_param = data_param_path + cfg['parameters_system'][var_number]
    param_df = pd.read_csv(file_path_param, index_col=1)
    param_value = param_df['value']

    # Import weather and demand data
    data = pd.read_csv(
        data_ts_path + cfg['time_series_file_name']).head(number_of_time_steps)

    # define costs function
    def ep_costs_f(capex, n, opex):
        ep_costs = (economics.annuity(capex, n, param_value['wacc'])
                    + capex * opex)
        return ep_costs
    ##################################

    # Initialise the energysystem
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq(cfg['frequenz'])
    date_time_index = data.index
    energysystem = solph.EnergySystem(timeindex=date_time_index)

    # Initiate the logger
    logger.define_logging(
        logfile='desalination_ro_{0}_{1}_{2}.log'.format(
            cfg['exp_number'], var_number, currentdate),
        logpath=results_path + '/logs',
        screen_level=logging.INFO,
        file_level=logging.DEBUG)

    #######################
    # Build up the system #
    #######################

    # busses

    bele = solph.Bus(label='electricity_bus')
    bwat = solph.Bus(label='water_bus')

    energysystem.add(bele, bwat)

    # sinks and sources
    pv = solph.Source(
        label='pv_source',
        outputs={bele: solph.Flow(
            fix=data['global_horinzontal_in_W_m2'],
            investment=solph.Investment(
                ep_costs=ep_costs_f(
                    param_value['pv_invest_costs_output_el'],
                    param_value['pv_lifetime'],
                    param_value['pv_opex'])))})

    demand = solph.Sink(
        label='water_demand',
        inputs={bwat: solph.Flow(
            fix=data['demand_water_in_qm'],
            nominal_value=1)})

    electr_output = solph.Sink(
        label='electric_output',
        inputs={bele: solph.Flow(
            nominal_value=11670842880000,
            summed_max=1,
            variable_costs=param_value['selling_price_electricity'])})

    energysystem.add(pv, demand, electr_output)

    # transformers

    desalination = solph.Transformer(
        label='desalination',
        inputs={bele: solph.Flow()},
        outputs={bwat: solph.Flow(
            min=param_value['min_desalination'],
            investment=solph.Investment(
                ep_costs=ep_costs_f(
                    param_value['desalination_ro_invest_costs'],
                    param_value['desalination_ro_lifetime'],
                    param_value['desalination_ro_opex']))
        )},
        conversion_factors={
            bele: param_value['desalination_ro_conv_factor_electric']})

    energysystem.add(desalination)

    # storages

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
        invest_relation_output_capacity=1/6,
        invest_relation_input_capacity=1/6,
        investment=solph.Investment(
            ep_costs=ep_costs_f(
                param_value['electric_stor_invest_costs_capacity'],
                param_value['electric_stor_lifetime'],
                param_value['electric_stor_opex'])))

    energysystem.add(storage_water, storage_electricity)

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
                    'desalination_ro_{0}_{1}_{2}.lp'.format(
                        cfg['exp_number'], var_number, currentdate))
        logging.info('Store lp-file in {0}.'.format(filename))
        model.write(filename, io_options={'symbolic_solver_labels': True})

    # save the results
    logging.info('Store the energy system with the results.')

    energysystem.results['main'] = processing.results(model)
    energysystem.results['meta'] = processing.meta_results(model)
    energysystem.results['param'] = (
        processing.parameter_as_dict(model))

    energysystem.dump(
        dpath=(results_path + '/dumps/' + '/'),
        filename='desalination_ro_{0}_{1}_{2}.oemof'.format(
            cfg['exp_number'], var_number, currentdate))
