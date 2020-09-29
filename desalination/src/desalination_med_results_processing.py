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
from datetime import datetime

# import oemof plots
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

df_all_var = pd.DataFrame()


def postprocessing_med(config_path, var_number):

    currentdate = datetime.today().strftime('%Y%m%d')

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # define the used directories
    abs_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
    results_path = abs_path + '/results'
    csv_path = results_path + '/optimisation_results/'
    plot_path = results_path + '/plots/'

    energysystem = solph.EnergySystem()
    energysystem.restore(
        dpath=(results_path + '/dumps/' + '/'),
        filename='desalination_med_{0}_{1}_{2}.oemof'.format(
            cfg['exp_number'], var_number, currentdate))

    # Look up investment costs. Therefor parameters must read again.

    file_path_param = abs_path + '/data/data_public/' +\
        cfg['parameters_system'][var_number]
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

    # scalars
    solar_scal = solph.views.node(
        energysystem.results['main'], 'solar_bus')['scalars']
    electricity_scal = solph.views.node(
        energysystem.results['main'], 'electricity_bus')['scalars']
    water_scal = solph.views.node(
        energysystem.results['main'], 'water_bus')['scalars']
    none_scal = solph.views.node(
        energysystem.results['main'], 'None')['scalars']
    none_scal_given = solph.views.node(
        energysystem.results['param'], 'None')['scalars']
    # thermal_high_scal = solph.views.node(
    #    energysystem.results['main'], 'thermal_high_bus')['scalars']
    # thermal_low_scal = solph.views.node(
    #    energysystem.results['main'], 'thermal_low_bus')['scalars']

    # sequences:
    df_seq_el = pd.DataFrame()
    df_seq_el['Electricity from power plant'] = \
        results_strings[('powerplant', 'electricity_bus')][
            'sequences'].flow
    df_seq_el['From electric storage'] = \
        results_strings[('storage_electricity', 'electricity_bus')][
            'sequences'].flow
    df_seq_el['Electricity to desalination'] = \
        results_strings[('electricity_bus', 'desalination')][
            'sequences'].flow
    df_seq_el['Electricity to collector'] = \
        results_strings[('electricity_bus', 'collector_transformer')][
            'sequences'].flow
    df_seq_el['Electrical output'] =\
        results_strings[('electricity_bus', 'electric_output')][
            'sequences'].flow
    df_seq_el['To electric storage'] = \
        results_strings[('electricity_bus', 'storage_electricity')][
            'sequences'].flow
    df_seq_el = df_seq_el.round(decimals=5)

    df_seq_wat = pd.DataFrame()
    df_seq_wat['Water from desalination'] = \
        results_strings[('desalination', 'water_bus')][
            'sequences'].flow
    df_seq_wat['From water storage'] = \
        results_strings[('storage_water', 'water_bus')][
            'sequences'].flow
    df_seq_wat['Water demand'] = \
        results_strings[('water_bus', 'water_demand')][
            'sequences'].flow
    df_seq_wat['To water storage'] = \
        results_strings[('water_bus', 'storage_water')][
            'sequences'].flow
    df_seq_wat = df_seq_wat.round(decimals=5)

    df_seq_sol = pd.DataFrame()
    df_seq_sol['Solar energy of the collector'] = \
        results_strings[('collector_source', 'solar_bus')][
            'sequences'].flow

    df_seq_thh = pd.DataFrame()
    df_seq_thh['Thermal energy from collector'] = \
        results_strings[('collector_transformer', 'thermal_high_bus')][
            'sequences'].flow
    df_seq_thh['From thermal storage'] = \
        results_strings[('storage_thermal', 'thermal_high_bus')][
            'sequences'].flow
    df_seq_thh['Thermal energy to power plant'] = \
        results_strings[('thermal_high_bus', 'powerplant')][
            'sequences'].flow
    df_seq_thh['High temperature thermal energy to desalination'] = \
        results_strings[('thermal_high_bus', 'auxiliary_transfomer')][
            'sequences'].flow
    df_seq_thh['To thermal storage'] = \
        results_strings[('thermal_high_bus', 'storage_thermal')][
            'sequences'].flow
    df_seq_thh = df_seq_thh.round(decimals=5)

    df_seq_thl = pd.DataFrame()
    df_seq_thl['Waste heat from power plant'] = \
        results_strings[('powerplant', 'thermal_low_bus')][
            'sequences'].flow
    df_seq_thl['Thermal energy from high temperature'] = \
        results_strings[('auxiliary_transfomer', 'thermal_low_bus')][
            'sequences'].flow
    df_seq_thl['Thermal energy to desalination'] = \
        results_strings[('thermal_low_bus', 'desalination')][
            'sequences'].flow
    df_seq_thl['Thermal energy excess'] = \
        results_strings[('thermal_low_bus', 'thermal_excess')][
            'sequences'].flow

    ## Costs ##

    # costs
    costs_total = energysystem.results['meta']['objective']
    # water
    water_sum = float(
        results_strings[('water_bus', 'water_demand')]['sequences'].sum())
    # costs per m3
    spec_costs = costs_total / water_sum

    ## storage usage ##
    storage_in_sum_el = float(
        results_strings[
            ('electricity_bus', 'storage_electricity')]['sequences'].sum())
    storage_in_sum_therm = float(
        results_strings[
            ('thermal_high_bus', 'storage_thermal')]['sequences'].sum())
    desalination_input_el = float(
        results_strings[
            ('electricity_bus', 'desalination')]['sequences'].sum())
    desalination_input_therm = float(
        results_strings[
            ('thermal_low_bus', 'desalination')]['sequences'].sum())
    storage_per_desal_input_el = storage_in_sum_el / desalination_input_el
    storage_per_desal_input_therm =\
        storage_in_sum_therm / desalination_input_therm

    ########################
    # Write results in csv #
    ########################

    # ## scalars ## #
    # base scalars:
    scalars_all = solar_scal\
        .append(electricity_scal)\
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
         pd.Series([costs_total], index=["costs per year"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([water_sum], index=["total water per year"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([spec_costs], index=["specific water costs"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([storage_per_desal_input_el],
                   index=["input storage per input desalination"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([storage_per_desal_input_therm],
                   index=["input storage per input desalination"])])

    # write scalars into csv for this experiment and variation
    scalars_all.to_csv(
        csv_path + 'desalination_med_{0}_{1}_{2}_scalars.csv'.format(
                        cfg['exp_number'], var_number, currentdate))

    # ## sequences ## #
    sequences_df = pd.concat([df_seq_sol, df_seq_el, df_seq_wat, df_seq_thh,
                              df_seq_thl], axis=1)
    sequences_df.to_csv(
        csv_path + 'desalination_med_{0}_{1}_{2}_sequences.csv'.format(
            cfg['exp_number'], var_number, currentdate))

    ####################
    # Plotting results #
    ####################

    # create plotting dataframes
    df_plot_el = df_seq_el
    df_plot_el[['Electricity to desalination', 'Electrical output',
                'Electricity to collector', 'To electric storage']] = \
        df_seq_el[['Electricity to desalination', 'Electrical output',
                   'Electricity to collector', 'To electric storage']] * -1
    df_plot_el_summer = df_plot_el[4345:4513]
    df_plot_el_winter = df_plot_el[4:172]

    df_plot_wat = df_seq_wat
    df_plot_wat[['To water storage', 'Water demand']] = \
        df_seq_wat[['To water storage', 'Water demand']] * -1
    df_plot_wat_summer = df_plot_wat[4345:4513]
    df_plot_wat_winter = df_plot_wat[4:172]

    df_plot_thh = df_seq_thh
    df_plot_thh[['Thermal energy to power plant',
                 'High temperature thermal energy to desalination',
                 'To thermal storage']] = \
        df_seq_thh[['Thermal energy to power plant',
                    'High temperature thermal energy to desalination',
                    'To thermal storage']] * -1
    df_plot_thh_summer = df_plot_thh[4345:4513]
    df_plot_thh_winter = df_plot_thh[4:172]

    df_plot_thl = df_seq_thl
    df_plot_thl[['Thermal energy to desalination',
                 'Thermal energy excess']] = \
        df_seq_thl[['Thermal energy to desalination',
                    'Thermal energy excess']] * -1
    df_plot_thl_summer = df_plot_thl[4345:4513]
    df_plot_thl_winter = df_plot_thl[4:172]

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
                       'darkred',
                       'lightgray',
                       'saddlebrown',
                       'black',
                       'darkorange',
                       'greenyellow',
                       'crimson',
                       'indianred',
                       'tan']

        col_list = []
        for k in keys:
            # electricity
            if k == 'Electricity to desalination':
                col_list.append(col_options[0])
            elif k == 'Electricity from power plant':
                col_list.append(col_options[9])
            elif k == 'Electricity to collector':
                col_list.append(col_options[1])
            elif k == 'To electric storage':
                col_list.append(col_options[3])
            elif k == 'From electric storage':
                col_list.append(col_options[4])
            elif k == 'Electrical output':
                col_list.append(col_options[10])

            # water
            elif k == 'Water demand':
                col_list.append(col_options[5])
            elif k == 'Water from desalination':
                col_list.append(col_options[0])
            elif k == 'From water storage':
                col_list.append(col_options[6])
            elif k == 'To water storage':
                col_list.append(col_options[7])

            # high thermal energy
            elif k == 'Thermal energy from collector':
                col_list.append(col_options[1])
            elif k == 'From thermal storage':
                col_list.append(col_options[7])
            elif k == 'Thermal energy to power plant':
                col_list.append(col_options[9])
            elif k == 'High temperature thermal energy to desalination':
                col_list.append(col_options[0])
            elif k == 'To thermal storage':
                col_list.append(col_options[8])

            # low thermal energy
            elif k == 'Thermal energy to desalination':
                col_list.append(col_options[0])
            elif k == 'Waste heat from power plant':
                col_list.append(col_options[9])
            elif k == 'Thermal energy from high temperature':
                col_list.append(col_options[1])
            elif k == 'Thermal energy excess':
                col_list.append(col_options[10])

        return col_list

    # ## create plots ## #
    # electricity
    color_list = make_color_list(df_plot_el_summer.keys())

    df_plot_el_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, Electricity', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        '../results/plots/' +
        'med_electricity_summer_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')

    df_plot_el_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, Electricity', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path +
                'med_electricity_winter_{0}_{1}_{2}.png'.format(
                    cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')

    # water
    color_list = make_color_list(df_plot_wat_summer.keys())

    df_plot_wat_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, Water', size=25)
    plt.ylabel('[m3/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path +
                'med_water_summer_{0}_{1}_{2}.png'.format(
                    cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')

    df_plot_wat_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, Water', size=25)
    plt.ylabel('[m3/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(30, 15)
    plt.savefig(plot_path +
                'med_water_winter_{0}_{1}_{2}.png'.format(
                    cfg['exp_number'], var_number, currentdate),
                dpi=150,
                bbox_inches='tight')

    # high temperature thermal energy
    color_list = make_color_list(df_plot_thh_summer.keys())

    df_plot_thh_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, high temperature thermal energy', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        plot_path +
        'med_high_temp_thermal_summer_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')

    df_plot_thh_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, high temperature thermal energy', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        plot_path +
        'med_high_temp_thermal_winter_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')

    # high temperature thermal energy
    color_list = make_color_list(df_plot_thl_summer.keys())

    df_plot_thl_summer.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, low temperature thermal energy', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        plot_path +
        'med_low_temp_thermal_summer_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')

    df_plot_thl_winter.plot(
        kind='area', stacked=True, color=color_list, linewidth=0)
    plt.title('MED, low temperature thermal energy', size=25)
    plt.ylabel('[Wh/h]', size=25)
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0,
               prop={'size': 25})
    plt.yticks(size=25)
    figure = plt.gcf()
    figure.set_size_inches(28, 15)
    plt.savefig(
        plot_path +
        'med_low_temp_thermal_winter_{0}_{1}_{2}.png'.format(
            cfg['exp_number'], var_number, currentdate),
        dpi=150,
        bbox_inches='tight')
