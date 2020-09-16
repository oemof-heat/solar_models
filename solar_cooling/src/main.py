# -*- coding: utf-8 -*-

"""
Date: 27,05,2020
Author: Franziska Pleissner
This App will model a cooling system for Oman. An earlier Version calculated
the results of 'Provision of cooling in Oman - a linear optimisation problem
with special consideration of different storage options' IRES 2019
This version is adapted for oemof 0.4 and uses the solar_thermal_collector from
the oemof thermal repository.
"""

from thermal_model import run_model_thermal
from thermal_results_processing import thermal_postprocessing
from electric_model import run_model_electric
from electric_results_processing import electric_postprocessing

import os
import yaml


def main(yaml_file):
    # Choose configuration file to run model with
    exp_cfg_file_name = yaml_file
    config_file_path = (
        os.path.abspath('../experiment_config/' + exp_cfg_file_name))
    with open(config_file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

#    global df_all_var

    if type(cfg['parameters_variation']) == list:
        scenarios = range(len(cfg['parameters_variation']))
    elif type(cfg['parameters_system']) == list:
        scenarios = range(len(cfg['parameters_system']))
    else:
        scenarios = range(1)

    for scenario in scenarios:
        if cfg['run_model_thermal']:
            run_model_thermal(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_model_electric']:
            run_model_electric(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_postprocessing_thermal']:
            thermal_postprocessing(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_postprocessing_electric']:
            electric_postprocessing(
                config_path=config_file_path,
                var_number=scenario)


main('experiment_0.yml')
