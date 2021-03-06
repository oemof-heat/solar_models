# -*- coding: utf-8 -*-
"""
Created on Dez 06 2018

@author: Franziska Pleissner

System C: concrete example: Plot of cooling process with a solar collector
"""

############
# Preamble #
############

# Import packages
import oemof.solph as solph
import oemof.solph.views as views
import oemof_visio as oev

import logging
import os
import yaml
import pandas as pd
from electric_model import ep_costs_func

# Import oemof plots
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

df_all_var = pd.DataFrame()


def electric_postprocessing(config_path, var_number):
    global df_all_var

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CLoader)

    # Define the used directories
    abs_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
    results_path = abs_path + '/results'
    csv_path = results_path + '/optimisation_results/'
    plot_path = results_path + '/plots/'

    energysystem = solph.EnergySystem()
    energysystem.restore(dpath=(results_path + '/dumps'),
                         filename='electric_model_{0}_{1}.oemof'.format(
                             cfg['exp_number'], var_number))

    sp = cfg['start_of_plot']
    ep = cfg['end_of_plot']

    # Look up investment costs. Therefor parameters must read again.
    if type(cfg['parameters_variation']) == list:
        file_path_param_01 = abs_path + '/data/data_public/' + cfg[
            'parameters_system']
        file_path_param_02 = abs_path + '/data/data_public/' + cfg[
            'parameters_variation'][var_number]
    elif type(cfg['parameters_system']) == list:
        file_path_param_01 = abs_path + '/data/data_public/' + cfg[
            'parameters_system'][var_number]
        file_path_param_02 = abs_path + '/data/data_public/' + cfg[
            'parameters_variation']
    else:
        file_path_param_01 = abs_path + '/data/data_public/' + cfg[
            'parameters_system']
        file_path_param_02 = abs_path + '/data/data_public/' + cfg[
            'parameters_variation']
    param_df_01 = pd.read_csv(file_path_param_01, index_col=1)
    param_df_02 = pd.read_csv(file_path_param_02, index_col=1)
    param_df = pd.concat([param_df_01, param_df_02], sort=True)
    param_value = param_df['value']

    logging.info('results received')

    #########################
    # Work with the results #
    #########################

    cool_bus = views.node(energysystem.results['main'], 'cool')
    waste_bus = views.node(energysystem.results['main'], 'waste')
    el_bus = views.node(energysystem.results['main'], 'electricity')
    ambient_res = views.node(energysystem.results['main'], 'ambient')
    none_res = views.node(energysystem.results['main'], 'None')

    # Sequences:
    cool_seq = cool_bus['sequences']
    waste_seq = waste_bus['sequences']
    el_seq = el_bus['sequences']
    ambient_seq = ambient_res['sequences']

    # Scalars
    cool_scal = cool_bus['scalars']
    waste_scal = waste_bus['scalars']
    el_scal = el_bus['scalars']
    none_scal = none_res['scalars']
    none_scal_given = views.node(
        energysystem.results['param'], 'None')['scalars']
    el_scal[(('pv', 'electricity'), 'invest')] = (
            el_scal[(('pv', 'electricity'), 'invest')]*param_value['size_pv'])
    # Conversion of the pv-investment-size, because Invest-object is normalized
    # at 0.970873786 kWpeak

    # solar fraction
    # electric:
    # control_el (No Power must go from grid to excess)
    df_control_el = pd.DataFrame()
    df_control_el['grid_el'] = el_seq[(('grid_el', 'electricity'), 'flow')]
    df_control_el['excess'] = el_seq[(('electricity', 'excess_el'), 'flow')]
    df_control_el['Product'] = (df_control_el['grid_el']
                                * df_control_el['excess'])

    el_from_grid = el_seq[(('grid_el', 'electricity'), 'flow')].sum()
    el_from_pv = el_seq[(('pv', 'electricity'), 'flow')].sum()
    el_to_excess = el_seq[(('electricity', 'excess_el'), 'flow')].sum()
    el_pv_used = el_from_pv - el_to_excess
    sol_fraction_el = el_pv_used / (el_pv_used + el_from_grid)

    # Power usage:
    el_used = el_seq[(('grid_el', 'electricity'), 'flow')].sum()

    # Power to the output:
    electricity_output = el_seq[(('electricity', 'excess_el'), 'flow')].sum()
    electricity_output_pv = el_seq[(('pv', 'electricity'), 'flow')].sum()

    # ## costs ## #

    costs_total = energysystem.results['meta']['objective']

    # Storage costs must be subtract for reference scenario or added
    # for the other scenarios.

    # reference scenario:
    if param_value['nominal_capacitiy_stor_el'] == 0:
        costs_total_wo_stor = (
            costs_total
            - (none_scal[(('storage_electricity', 'None'), 'invest')]
                * none_scal_given[
                    (('storage_electricity', 'None'), 'investment_ep_costs')])
            - (none_scal[(('storage_cool', 'None'), 'invest')]
                * none_scal_given[
                    (('storage_cool', 'None'), 'investment_ep_costs')]))
    # other scenarios:
    else:
        # calculation of ep_costs
        ep_costs_el_stor = ep_costs_func(
            param_value['invest_costs_stor_el_capacity'],
            param_value['lifetime_stor_el'],
            param_value['opex_stor_el'],
            param_value['wacc'])
        ep_costs_cool_stor = ep_costs_func(
            param_value['invest_costs_stor_cool_capacity'],
            param_value['lifetime_stor_cool'],
            param_value['opex_stor_cool'],
            param_value['wacc'])
        # calculation of the scenario costs inclusive storage costs
        costs_total_w_stor = (
                costs_total
                + (none_scal_given[
                        (('storage_cool', 'None'), 'nominal_capacity')]
                   * ep_costs_cool_stor)
                + (none_scal_given[
                        (('storage_electricity', 'None'), 'nominal_capacity')]
                   * ep_costs_el_stor))

    ########################
    # Write results in csv #
    ########################

    # ## scalars ## #
    # base scalars:
    scalars_all = cool_scal\
        .append(waste_scal)\
        .append(el_scal)\
        .append(none_scal)
    for i in range(0, none_scal_given.count()):
        if 'nominal_capacity' in none_scal_given.index[i]:
            scalars_all = pd.concat(
                [scalars_all,
                 pd.Series([none_scal_given[i]],
                           index=[none_scal_given.index[i]])])

    # solar fractions
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([sol_fraction_el],
                   index=["('solar fraction', 'electric'), ' ')"])])
    if df_control_el['Product'].sum() != 0:
        scalars_all = pd.concat(
            [scalars_all,
             pd.Series([df_control_el['Product'].sum()],
                       index=["Has to be 0!!!"])])

    # various results
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([el_used],
                   index=["('grid_el', 'electricity'), 'summe')"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([electricity_output],
                   index=["('electricity', 'output'), 'summe')"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([electricity_output_pv],
                   index=["('pv', 'electricity'), 'summe')"])])

    # costs with or without storage (depends on reference scenario or not)
    if param_value['nominal_capacitiy_stor_el'] != 0:
        scalars_all = pd.concat(
            [scalars_all,
             pd.Series([costs_total_w_stor],
                       index=["('costs', 'w_stor'), 'per year')"])])
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series([costs_total],
                   index=["('costs', 'wo_stor'), 'per year')"])])
    if param_value['nominal_capacitiy_stor_el'] == 0:
        scalars_all = pd.concat(
            [scalars_all,
             pd.Series([costs_total_wo_stor],
                       index=["('costs', 'wo stor'), 'per year')"])])

    # experiment number and variation
    scalars_all = pd.concat(
        [scalars_all,
         pd.Series(['{0}_{1}'.format(cfg['exp_number'], var_number)],
                   index=["('Exp', 'Var'), 'number')"])])

    # write scalars into csv for this experiment and variation
    scalars_all.to_csv(
        csv_path + 'electric_model_{0}_{1}_scalars.csv'.format(
            cfg['exp_number'], var_number),
        header=False)

    # write scalars for all variations of the experiment into csv
    df_all_var = pd.concat([df_all_var, scalars_all], axis=1, sort=True)
    if var_number == (cfg['number_of_variations']-1):
        df_all_var.to_csv(
            csv_path
            + 'electric_model_{0}_scalars_all_variations.csv'.format(
                cfg['exp_number']))
        logging.info('Writing the results for all variations into csv')

    # ## sequences ## #
    sequences_df = pd.merge(ambient_seq, waste_seq, left_index=True,
                            right_index=True)
    sequences_df = pd.merge(sequences_df, el_seq, left_index=True,
                            right_index=True)
    sequences_df = pd.merge(sequences_df, cool_seq,
                            left_index=True, right_index=True)
    sequences_df.to_csv(
        csv_path + 'electric_model_{0}_{1}_sequences.csv'.format(
            cfg['exp_number'], var_number),
        header=False)

    ########################
    # Plotting the results # # to adapt for the use case
    ########################

    cool_seq_resample = cool_seq.iloc[sp:ep]
    waste_seq_resample = waste_seq.iloc[sp:ep]
    el_seq_resample = el_seq.iloc[sp:ep]
    ambient_seq_resample = ambient_seq.iloc[sp:ep]

    def shape_legend(node, reverse=False, **kwargs):  # just copied
        handels = kwargs['handles']
        labels = kwargs['labels']
        axes = kwargs['ax']
        parameter = {}

        new_labels = []
        for label in labels:
            label = label.replace('(', '')
            label = label.replace('), flow)', '')
            label = label.replace(node, '')
            label = label.replace(',', '')
            label = label.replace(' ', '')
            new_labels.append(label)
        labels = new_labels

        parameter['bbox_to_anchor'] = kwargs.get('bbox_to_anchor', (1, 1))
        parameter['loc'] = kwargs.get('loc', 'upper left')
        parameter['ncol'] = kwargs.get('ncol', 1)
        plotshare = kwargs.get('plotshare', 0.9)

        if reverse:
            handels = handels.reverse()
            labels = labels.reverse()

        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * plotshare, box.height])

        parameter['handles'] = handels
        parameter['labels'] = labels
        axes.legend(**parameter)
        return axes

    cdict = {
        (('storage_cool', 'cool'), 'flow'): '#555555',
        (('cool', 'storage_cool'), 'flow'): '#9acd32',
        (('cool', 'demand'), 'flow'): '#cd0000',
        (('grid_el', 'electricity'), 'flow'): '#999999',
        (('pv', 'electricity'), 'flow'): '#ffde32',
        (('storage_electricity', 'electricity'), 'flow'): '#9acd32',
        (('electricity', 'storage_electricity'), 'flow'): '#9acd32',
        (('electricity', 'compression_chiller'), 'flow'): '#4682b4',
        (('electricity', 'cooling_tower'), 'flow'): '#ff0000',
        (('storage_cool', 'None'), 'capacity'): '#555555',
        (('storage_cool', 'cool'), 'flow'): '#9acd32',
        (('compression_chiller', 'waste'), 'flow'): '#4682b4',
        (('electricity', 'excess_el'), 'flow'): '#999999',
        (('waste', 'cool_tower'), 'flow'): '#42c77a'}

    # define order of inputs and outputs
    inorderel = [(('pv', 'electricity'), 'flow'),
                 (('storage_electricity', 'electricity'), 'flow'),
                 (('grid_el', 'electricity'), 'flow')]
    outorderel = [(('electricity', 'compression_chiller'), 'flow'),
                  (('electricity', 'cooling_tower'), 'flow'),
                  (('electricity', 'storage_electricity'), 'flow'),
                  (('electricity', 'excess_el'), 'flow')]

    fig = plt.figure(figsize=(15, 15))

    # plot electrical energy
    my_plot_el = oev.plot.io_plot('electricity', el_seq_resample, cdict=cdict,
                                  inorder=inorderel, outorder=outorderel,
                                  ax=fig.add_subplot(2, 2, 1), smooth=False)

    ax_el = shape_legend('electricity', **my_plot_el)

    ax_el.set_ylabel('Power in kW')
    ax_el.set_xlabel('time')
    ax_el.set_title("results of the electric model - electricity flows")

    plt.savefig(
        plot_path + 'electric_model_results_plot_{0}_{1}.png'.format(
            cfg['exp_number'], var_number))
    csv_plot = pd.merge(el_seq_resample, cool_seq_resample,
                        left_index=True, right_index=True)
    csv_plot = pd.merge(csv_plot, el_seq_resample,
                        left_index=True, right_index=True)
    csv_plot.to_csv(
        plot_path + 'electric_model_results_plot_data_{0}_{1}.csv'.format(
            cfg['exp_number'], var_number))

    return df_all_var
