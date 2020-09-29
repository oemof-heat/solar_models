# -*- coding: utf-8 -*-

"""
Date: 23.06.2020
Author: Franziska Pleissner
This App will model the energy supply of a seawater desalination process. A
concentrating solar power plant is used to provide heat and electricity to the
desalination facility. The latter is not considered in detail, but is#
represented as a transfomer with a fixed efficiency.
"""

from desalination_med_model import run_model_med
from desalination_ro_model import run_model_ro
from desalination_med_results_processing import postprocessing_med
from desalination_ro_results_processing import postprocessing_ro
from desalination_precalculation import run_precalculation

import os
import yaml


def main(yaml_file):
    # Choose configuration file to run model with
    exp_cfg_file_name = yaml_file
    config_file_path = (
        os.path.abspath('../experiment_config/' + exp_cfg_file_name))
    with open(config_file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if type(cfg['parameters_system']) == list:
        scenarios = range(len(cfg['parameters_system']))
    else:
        scenarios = range(1)

    for scenario in scenarios:
        if cfg['run_precalculation']:
            run_precalculation(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_model_med']:
            run_model_med(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_postprocessing_med']:
            postprocessing_med(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_model_ro']:
            run_model_ro(
                config_path=config_file_path,
                var_number=scenario)
        if cfg['run_postprocessing_ro']:
            postprocessing_ro(
                config_path=config_file_path,
                var_number=scenario)


main('experiment_desalination_0.yml')
# main('experiment_desalination_3.yml')
# main('experiment_desalination_4.yml')
