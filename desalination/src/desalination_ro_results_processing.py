# -*- coding: utf-8 -*-
"""

Date: 10.07.2020
Author: Franziska Pleissner

System C: deslination med: processing of the results
"""

############
# Preamble #
############

# Import packages
from oemof import solph
import oemof_visio as oev

from oemof.tools import logger, economics

import logging
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# import oemof plots
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

df_all_var = pd.DataFrame()


def postprocessing_ro(config_path, var_number):

    currentdate = datetime.today().strftime('%Y%m%d')

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # define the used directories
    abs_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
    results_path = abs_path + '/results'
    csv_path = results_path + '/optimisation_results/'
    plot_path = results_path + '/plots/'

    energysystem = solph.EnergySystem()
    energysystem.restore(
        dpath=(results_path + '/dumps/' + '/'),
        filename='desalination_ro_{0}_{1}_{2}.oemof'.format(
            cfg['exp_number'], var_number, currentdate))

    # Look up investment costs. Therefor parameters must read again.

    file_path_param = abs_path +\
        '/data/data_public/' + cfg['parameters_system'][var_number]
    param_df = pd.read_csv(file_path_param, index_col=1)
    param_value = param_df['value']

    logging.info('results received')

    def ep_costs_f(capex, n, opex):
        ep_costs = (economics.annuity(capex, n, param_value['wacc'])
                    + capex * opex)
        return ep_costs

    #########################
    # Work with the results #
    #########################

    results_strings = (
        solph.views.convert_keys_to_strings(energysystem.results['main']))

    # ### scalars
    electricity_scal = solph.views.node(
        energysystem.results['main'], 'electricity_bus')['scalars']
    water_scal = solph.views.node(
        energysystem.results['main'], 'water_bus')['scalars']
    none_scal = solph.views.node(
        energysystem.results['main'], 'None')['scalars']
    none_scal_given = solph.views.node(
        energysystem.results['param'], 'None')['scalars']

    # ### sequences
    # electricity
    df_seq_el = pd.DataFrame()
    df_seq_el['PV'] =\
        results_strings[('pv_source', 'electricity_bus')][
            'sequences'].flow
    df_seq_el['From electric storage'] =\
        results_strings[('storage_electricity', 'electricity_bus')][
            'sequences'].flow
    df_seq_el['Electricity to desalination'] =\
        results_strings[('electricity_bus', 'desalination')][
            'sequences'].flow
    df_seq_el['Electrical output'] =\
        results_strings[('electricity_bus', 'electric_output')][
            'sequences'].flow
    df_seq_el['To electric storage'] =\
        results_strings[('electricity_bus', 'storage_electricity')][
            'sequences'].flow
    df_seq_el = df_seq_el.round(decimals=5)

    # water
    df_seq_wat = pd.DataFrame()
    df_seq_wat['Water from desalination'] = \
        results_strings[
            ('desalination', 'water_bus')]['sequences'].flow
    df_seq_wat['From water storage'] = \
        results_strings[
            ('storage_water', 'water_bus')]['sequences'].flow
    df_seq_wat['Water demand'] = \
        results_strings[
            ('water_bus', 'water_demand')]['sequences'].flow
    df_seq_wat['To water storage'] = \
        results_strings[
            ('water_bus', 'storage_water')]['sequences'].flow
    df_seq_wat = df_seq_wat.round(decimals=5)

    # ### processed values

    ## costs
    # costs
    costs_total = energysystem.results['meta']['objective']
    # water
    water_sum = float(
        results_strings[('water_bus', 'water_demand')]['sequences'].sum())
    # costs per qm
    spec_costs = costs_total / water_sum

    ## storage usage
    storage_in_sum = float(
        results_strings[
            ('electricity_bus', 'storage_electricity')]['sequences'].sum())
    desalination_input = float(
        results_strings[('electricity_bus', 'desalination')][
            'sequences'].sum())
    pv_sum = float(
        results_strings[('pv_source', 'electricity_bus')]['sequences'].sum())
    storage_per_desal_input = storage_in_sum / desalination_input
    pv_per_invest = pv_sum / float(
        results_strings[('pv_source', 'electricity_bus')]['scalars'])

    ########################
    # Write results in csv #
    ########################

    # ## scalars ## #
    # base scalars:
    scalars_all = electricity_scal\
        .append(water_scal)\
        .append(none_scal)
    for i in range(0, none_scal_given.count()):
        if 'nominal_capacity' in none_scal_given.index[i]:
            scalars_all = pd.concat(
                [scalars_all,
                 pd.Series([none_scal_given[i]],
                           index=[none_scal_given.index[i]])])

    # other values
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([costs_total], index=["('costs', 'None'), 'per year')"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([water_sum], index=["total water per year"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([spec_costs], index=["specific water costs"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([storage_per_desal_input],
                   index=["input storage per input desalination"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([pv_per_invest], index=["pv output per Wpeak"])])

    # write scalars into csv
    scalars_all.to_csv(
        csv_path + 'desalination_ro_{0}_{1}_{2}_scalars.csv'.format(
                        cfg['exp_number'], var_number, currentdate))

    # ## sequences ## #
    sequences_df = pd.concat([df_seq_el, df_seq_wat], axis=1)

    sequences_df.to_csv(
        csv_path + 'desalination_ro_{0}_{1}_{2}_sequences.csv'.format(
            cfg['exp_number'], var_number, currentdate))

    ####################
    # Plotting results #
    ####################

    # create plotting dataframes
    df_plot_el = df_seq_el
    df_plot_el[['Electricity to desalination',
                'Electrical output', 'To electric storage']] = \
        df_seq_el[['Electricity to desalination',
                   'Electrical output', 'To electric storage']] * -1

    df_plot_el_summer = df_plot_el[4345:4513]
    df_plot_el_winter = df_plot_el[4:172]

    df_plot_water = df_seq_wat
    df_plot_water[['To water storage', 'Water demand']] = \
        df_seq_wat[['To water storage', 'Water demand']] * -1

    df_plot_water_summer = df_plot_water[4345:4513]
    df_plot_water_winter = df_plot_water[4:172]

    def make_color_list(keys):
        """Return list with colors for plots sorted by keys to make sure each
        component/technology appears in the same color in every plot
        (improves recognition)."""
        # Define colors
        col_options = ['darkblue',
                       'gold',
                       'darkgray',
                       'darkgreen',
                       'lightgreen',
                       'lightskyblue',
                       'blueviolet',
                       'fuchsia',
                       'black',
                       'darkred',
                       'darkorange',
                       'greenyellow',
                       'crimson',
                       'saddlebrown',
                       'indianred',
                       'lightgray',
                       'tan']

        col_list = []
        for k in keys:
            # Strom
            if k == 'Electricity to desalination':
                col_list.append(col_options[0])
            elif k == 'PV':
                col_list.append(col_options[1])
            elif k == 'Electrical output':
                col_list.append(col_options[2])
            elif k == 'To electric storage':
                col_list.append(col_options[3])
            elif k == 'From electric storage':
                col_list.append(col_options[4])

            # Wasser
            elif k == 'Water demand':
                col_list.append(col_options[5])
            elif k == 'Water from desalination':
                col_list.append(col_options[0])
            elif k == 'From water storage':
                col_list.append(col_options[6])
            elif k == 'To water storage':
                col_list.append(col_options[7])
        return col_list

    # ## create plots ## #
    # electricity
    color_list = make_color_list(df_plot_el_summer.keys())

    df_plot_el_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('RO, Electricity', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        plot_path + 'ro_electricity_winter_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')

    df_plot_el_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('RO, Electricity', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path + 'ro_electricity_summer_{0}_{1}_{2}.png'.format(
                    cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')
    # water
    color_list = make_color_list(df_plot_water_summer.keys())

    df_plot_water_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('RO, Water', size=25)
    plt.ylabel('[m3/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path + 'ro_water_summer_{0}_{1}_{2}.png'.format(
        cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')

    df_plot_water_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('RO, Water', size=25)
    plt.ylabel('[m3/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path + 'ro_water_winter_{0}_{1}_{2}.png'.format(
        cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')
