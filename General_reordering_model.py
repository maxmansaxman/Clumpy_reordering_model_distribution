''' Command-line based model for reordering calcite and dolomite by solid state exchange'''

import ClumpyCool_func as cc
#import sys
# sys.path.insert(0, '/Users/Max/Github/clumpy')
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('pdf')
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import matplotlib.patches as patches
import os
#import random
#from scipy import optimize
#import pylab
# import CIDS_func as cf
#from scipy.integrate import odeint
#from scipy.optimize import leastsq
from scipy.interpolate import PchipInterpolator
#plt.style.use('ggplot')
plt.close('all')
plt.ion()

def fit_and_plot(T_t_points):
    ''' fn to fit the Temp-time array, and plot it up'''
    model_resoluion = int(1e4)
    T_t_fn = PchipInterpolator(T_t_points['time_yrs'], T_t_points['temp_C'])
    time_array = np.linspace(0, T_t_points['time_yrs'].max(), num = model_resoluion)
    fig, ax = plt.subplots()
    ax.plot(T_t_points['time_yrs'], T_t_points['temp_C'], 's', color = 'C3', label = 'Fixed points')
    ax.plot(time_array, T_t_fn(time_array), '-', color = 'C4', label = 'T-t fit')
    ax.legend(loc = 'best')
    ax.set_xlabel('Time (yrs)')
    ax.set_ylabel(r'Temp (°C)')
    plt.show()
    fig.savefig('Model_T_t_path.pdf')
    return(time_array, T_t_fn)

def plot_solutions(T_t_points, T_t_fn, time_array, D47_pred, sigma_choice = 1.0, pairs_plot = False):
    ''' fin to plot the results from the model'''
    fig, ax = plt.subplots(2, figsize = (6,8))
    fig1, ax1 = plt.subplots()

    ax[0].plot(T_t_points['time_yrs'], T_t_points['temp_C'], 's', color = 'C3', label = 'Fixed points')
    ax[0].plot(time_array, T_t_fn(time_array), '-', color = 'C4', label = 'T-t fit')
    ax1.plot(T_t_points['time_yrs'], T_t_points['temp_C'], 's', color = 'C3', label = 'Fixed points')
    ax1.plot(time_array, T_t_fn(time_array), '-', color = 'C4', label = 'T-t fit')
    ax[1].plot(T_t_points['time_yrs'], cc.temp_to_D47_ARF(T_t_points['temp_C']), 's', color = 'C3', label = 'Fixed points')
    ax[1].plot(time_array, cc.temp_to_D47_ARF(T_t_fn(time_array)), '-', color = 'C4', label = 'T-t fit')

    color_dict = {'dolomite': 'C1', 'calcite': 'C0'}
    for key in D47_pred.keys():
        run_steps = D47_pred[key].shape[0]
        run_jumps = int(run_steps/1000)
        ax[0].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred'], '-', color = color_dict[key],label = '{0} model'.format(key))
        ax[0].fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred_upper'], D47_pred[key].loc[::run_jumps,'Temp_pred_lower'], color = color_dict[key], alpha = 0.5 )
#        ax[1].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred'].apply(cc.temp_to_D47_ARF), '-', color = color_dict[key],label = '{0} model'.format(key))
#        ax[1].fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred_upper'].apply(cc.temp_to_D47_ARF), D47_pred[key].loc[::run_jumps,'Temp_pred_lower'].apply(cc.temp_to_D47_ARF), color = color_dict[key], alpha = 0.5 )
        ax[1].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'D47_pred'], '-', color = color_dict[key],label = '{0} model'.format(key))
        ax[1].fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'D47_pred_upper'], D47_pred[key].loc[::run_jumps,'D47_pred_lower'], color = color_dict[key], alpha = 0.5 )
        ax1.plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred'], '-', color = color_dict[key],label = '{0} model'.format(key))
        ax1.fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred_upper'], D47_pred[key].loc[::run_jumps,'Temp_pred_lower'], color = color_dict[key], alpha = 0.5 )

    ax1.legend(loc = 'upper left')
    ax1.set_xlabel('Time (yrs)')
    ax1.set_ylabel(r'Temp $^{\circ}$C')
    ax1.text(ax1.get_xlim()[1]-(ax1.get_xlim()[1]-ax1.get_xlim()[0])/8.0, ax1.get_ylim()[0]+(ax1.get_ylim()[1]-ax1.get_ylim()[0])/20.0, r'{0}$\sigma$ err bars'.format(sigma_choice), size = 'small', alpha = 0.5)

    ax[0].legend(loc = 'best')
    ax[1].set_xlabel('Time (yrs)')


    ax[0].set_ylabel(r'Temp $^{\circ}$C')
    ax[1].set_ylabel(r'$\Delta_{47,\regular{CDES25}} $ (‰)')
    ax[1].text(ax[1].get_xlim()[1]-(ax[1].get_xlim()[1]-ax[1].get_xlim()[0])/8.0, ax[1].get_ylim()[0]+(ax[1].get_ylim()[1]-ax[1].get_ylim()[0])/20.0, r'{0}$\sigma$ err bars'.format(sigma_choice), size = 'small', alpha = 0.5)

    if pairs_plot:
        fig2, ax2 = plt.subplots(2, figsize = (6,8))
        ax2[0].plot(T_t_points['time_yrs'], T_t_points['temp_C'], 's', color = 'C3', label = 'Fixed points')
        ax2[0].plot(time_array, T_t_fn(time_array), '-', color = 'C4', label = 'T-t fit')
#        ax2[1].plot(time_array, T_t_fn(time_array), '-', color = 'C4', label = 'T-t fit')

        for key in D47_pred.keys():
            run_steps = D47_pred[key].shape[0]
            run_jumps = int(run_steps/1000)
            ax2[0].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred'], '-', color = color_dict[key],label = '{0} model'.format(key))
            ax2[0].fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'Temp_pred_upper'], D47_pred[key].loc[::run_jumps,'Temp_pred_lower'], color = color_dict[key], alpha = 0.5 )
            ax2[1].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'d_pairs_equil'], '-', color = 'C4',label = '{0} pairs equil'.format(key))
            ax2[1].plot(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'d_pairs_pred'], '-', color = color_dict[key],label = '{0} pairs predicted'.format(key))
            ax2[1].fill_between(D47_pred[key].loc[::run_jumps,'time_yrs'], D47_pred[key].loc[::run_jumps,'d_pairs_pred_upper'], D47_pred[key].loc[::run_jumps,'d_pairs_pred_lower'], color = color_dict[key], alpha = 0.5 )
        ax2[0].set_ylabel(r'Temp $^{\circ}$C')
        ax2[1].set_xlabel(r'Time (Ma)')
        ax2[1].set_ylabel(r'$\delta \regular{[pairs]}$ (‰)')
#        ax2[1].set_ylim(ax2.get_ylim()[::-1])
        ax2[1].legend(loc = 'best')
        ax2[0].legend(loc = 'best')
        fig2.savefig('Model_predictions_with_pairs.pdf')

    plt.show()
    fig.savefig('Model_predictions.pdf')
    fig1.savefig('Model_predictions_justT.pdf')

    return

def save_solutions(D47_pred, path_name):
    ''' fn to save model outputs into excel sheet'''
    print('Saving outputs... ')
    columns_to_save = ['time_yrs', 'Temp_true','D47_pred','D47_pred_upper', 'D47_pred_lower', 'd_pairs_equil', 'd_pairs_pred', 'd_pairs_pred_upper', 'd_pairs_pred_lower', 'Temp_pred', 'Temp_pred_upper', 'Temp_pred_lower']
    for key in D47_pred.keys():
        run_steps = D47_pred[key].shape[0]
        run_jumps = int(run_steps/1000)
        D47_pred[key].loc[::run_jumps,columns_to_save].to_excel('{0}_model_{1}.xlsx'.format(path_name.strip('.xlsx').strip('.xls'), key))
    print('Done! ')
    return

print('Welome to the ClumpyCool carbonate clumped isotope reordering model')
while True:
    print('First, need a temperature--time path')
    task_choice = input(' (I)mport excel sheet? \n (D)raw on-screen?  \n (Q)uit \n ').upper()
    if task_choice == 'I':
        print('Importing an excel sheet \n First column: time (in years) \n Second column: Temperature (in C) ')
        while True:
            path_name = input('Drag an excel spreadsheet with the T-t path constraints: ').strip()
            path_name = path_name.strip("'").strip('"')
            path_name = os.path.abspath(path_name)
            acceptable_file_types = ('.xls', '.xlsx')
            if os.path.exists(path_name):
                if path_name.endswith(acceptable_file_types):
                    print('Reading T-t path.. ')
                    # If looks like an excel file, import it
                    T_t_points = pd.read_excel(path_name)
                    # rename first two columns, because we don't know what those came in as
                    T_t_points.rename(columns= {T_t_points.columns[0]: 'time_yrs', T_t_points.columns[1]: 'temp_C'}, inplace = True)
                    print('Drawing imported T-t path... ')
                    time_array, T_t_fn = fit_and_plot(T_t_points)
                    accept_path = input(' Accept T-t path? \n (Y)es or (N)o... ').upper()
                    if accept_path in ['Y', '']:
                        print('Using this path ')
                        model_choices = ['s', 'p']
                        while True:
                            model_choice_letter = input(' Use (S)tolper & Eiler (2015) model \n or (P)assey & Henkes (2012) model? ').lower()
                            if model_choice_letter in model_choices:
                                break
                            print('Invalid choice of model, please try again')
                        mineral_choice_dict = {'B': 'both', 'D': 'dolomite', 'C': 'calcite'}
                        while True:
                            mineral_choice_letter = input(' Model (d)olomite, (c)alcite, or (b)oth? ').upper()
                            if mineral_choice_letter in mineral_choice_dict.keys():
                                break
                            print('Invalid choice of mineral, please try again')
                        while True:
                            sigma_choice_str = input(' Choice of sigma error envelope? (default: 1.0) ')
                            try:
                                sigma_choice = np.float(sigma_choice_str)
                                if sigma_choice <= 5.0:
                                    break
                                print('Invalid sigma choice: {0} is greater than 5.0'.format(sigma_choice))
                            except(ValueError):
                                sigma_choice = 1.0
                                print('No sigma choice given, going with default (1.0)')
                                break
                        print('\n Running models now...')
                        D47_pred = cc.model_wrapper_with_prompts(time_array, T_t_fn,
                        mineral_choice = mineral_choice_dict[mineral_choice_letter],
                        model_choice = model_choice_letter,
                        sigma_choice = sigma_choice)
                        print('\n Modeling complete. \n Plotting solutions now...')
                        if model_choice_letter == 's':
                            plot_solutions(T_t_points, T_t_fn, time_array, D47_pred, sigma_choice = sigma_choice, pairs_plot = True)
                        else:
                            plot_solutions(T_t_points, T_t_fn, time_array, D47_pred, sigma_choice = sigma_choice, pairs_plot = False)
                        print('\n Plotting complete. \n Now saving data...')
                        save_solutions(D47_pred, path_name)
                        print('\n Data save complete. ')
                        break

                else:
                    print('Invalid file type, must end with: ')
                    print([i for i in acceptable_file_types])
            else:
                print('Nonexistent file, please try again ')

        break

    if task_choice == 'D':
        print('Selecting points on screen ')
        print('Still working on this...')

    if task_choice == 'Q':
        print('Goodbye! ')
        break
    else:
        pass
