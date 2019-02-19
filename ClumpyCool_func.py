''' General model for modeling clumped isotope reordering. can change between mineral,
model type, bulk composition, etc'''


import numpy as np
import pandas as pd
import random
from scipy import optimize
from scipy.integrate import odeint
from scipy.interpolate import PchipInterpolator


def temp_to_D47_ARF(T_C):
    '''Function to apply a temperature calibration'''
    # Schauble 2006 4th-order polynomial fit to calcite. Dolomite is undetectably different, so just use calcite
#    A = -3.40752e6
#    B = 2.36545e4
#    C = -2.63167
#    D = -5.85372e-3
    T_K = T_C+273.15
#    K_eq = A/(T_K**4) + B/(T_K**3) + C/(T_K**2) + D/T_K + 1
#
#    D63_pred = (K_eq-1)*1000
#    # empirical observation of kinetic isotope effect between D63 and D47_ARF_90C, from Bonifacie et al., 2017
#    D47_ARF_90 = D63_pred + 0.176
#    D47_ARF_acid = D47_ARF_90 + 0.092
#    return(D47_ARF_acid)
    return((-3.40752e6/(T_K**4) + 2.36545e4/(T_K**3) - 2.63167/(T_K**2) - 5.85372e-3/T_K)*1000 + 0.176 + 0.092)

def D47_to_temp_ARF_long(D47_ARF_acid):
    '''Function to apply a temperature calibration'''
    # Schauble 2006 4th-order polynomial fit to calcite. Dolomite is undetectably different, so just use calcite
    D47_params = (D47_ARF_acid,)
    T_C_guess = 150
    T_C_res = optimize.minimize(D47_temp_minimization_func_schauble, T_C_guess, args = D47_params, method = 'Nelder-Mead', tol = 1.0)
    # T_C_res = optimize.minimize(D47_temp_minimization_func_schauble, T_C_guess, args = D47_params)

    return(T_C_res.x[0])

def D47_to_temp_ARF(D47_ARF_acid):
    '''Function to apply a temperature calibration'''
    # Schauble 2006 4th-order polynomial fit to calcite. Dolomite is undetectably different, so just use calcite
    # Using newton-raphson method to rapidly converge on root of 4th-order polynomial fn
    T_K_guess = 273.15
    D47_ARF_90 = D47_ARF_acid-0.092
    A = -3.40752e9
    B = 2.36545e7
    C = -2.63167e3
    D = -5.85372
    T_K = T_K_guess
    # D47_pred = A/(T_K**4) + B/(T_K**3) + C/(T_K**2) + D/T_K + 0.176
    for i in range(5):
        T_K -= ((A/(T_K**4) + B/(T_K**3) + C/(T_K**2) + D/T_K + 0.176)-D47_ARF_90)/(-4*A/(T_K**5) - 3*B/(T_K**4) - 2*C/(T_K**3) - D/(T_K**2))
    # empirical observation of kinetic isotope effect between D63 and D47_ARF_90C, from Bonifacie et al., 2017
    # residual = np.abs(D47_pred - (D47_ARF - 0.092))
    T_C = T_K - 273.15
    return(T_C)

def D47_temp_minimization_func_schauble(T_C, D47_ARF,):
    ''' Function to find the T (in K) that best fits the Schauble et al. (2006) eqn'''
    A = -3.40752e6
    B = 2.36545e4
    C = -2.63167
    D = -5.85372e-3
    T_K = T_C+273.15
    K_eq = A/(T_K**4) + B/(T_K**3) + C/(T_K**2) + D/T_K + 1
    D63_pred = (K_eq-1)*1000
    # empirical observation of kinetic isotope effect between D63 and D47_ARF_90C, from Bonifacie et al., 2017
    D47_pred = D63_pred + 0.176
    residual = np.abs(D47_pred - (D47_ARF - 0.092))
    return(residual)

def plot_T_scale(axis):
    ''' Plots a secondary y-axis with temperature scale to match the D47 on ARF '''
    # 1. Get D47 labels
    # tlabels = axis.get_yticklabels(which = 'both')
    # 2. For some reason, have to save fig first
    fig = plt.gcf()
    fig.savefig('temp.pdf')
    # copy the old axis
    axis2 = axis.twinx()
    axis2.grid(False)
    # ensure that they align
    axis2.set_ylim(axis.get_ylim())
    # Now, extract ticklabels from new axis
    tlabels = axis.get_yticklabels(which = 'both')
    y2_temps = []
    for i in range(len(tlabels)):
        tl = tlabels[i]
        D47_str = tl.get_text()
        if len(D47_str) > 0:
            if str(D47_to_temp_ARF(float(D47_str))) == 'nan':
                y2_temps.append('')
            else:
                y2_temps.append('{0:.0f}'.format(np.around(D47_to_temp_ARF(float(D47_str)),0)))
        else:
            y2_temps.append('')
    # axis2 = axis.twinx()
    # axis2.grid(False)
    axis2.set_ylabel(r'Temperature ($^{\circ}$C)')
    axis2.set_yticklabels(y2_temps)
    return

def calculate_resolution_tolerances(times_yrs, time_temp_fn):
    temp_array = time_temp_fn(times_yrs)
    D47_array = temp_to_D47_ARF(temp_array)
    relative_delta = np.abs((D47_array[1:]-D47_array[0:-1]))*(temp_array[1:]**2)
#    relative_delta = np.insert(relative_delta, 0,relative_delta[0])
#    relative_delta = np.insert(relative_delta, 0,relative_delta[0])

#    tolerance_array = np.ones(len(times_yrs)-1)*1e-14
#    tolerance_array[relative_delta>2] = 1e-15
#    tolerance_array[relative_delta>4] = 1e-17
    tolerance = 1e-14
    if relative_delta.max()>2:
        if relative_delta.max()>4:
            tolerance = 1e-17
        else:
            tolerance = 1e-15
    return(tolerance)


def predict_D47(times_yrs, time_temp_fn, D47_init, model = 's', bulk_comp = [0,0], mineral = 'calcite', T_soak_choice = False, sigma_choice = 1):
    if model == 's':
        # actual weighted values, fit pairs
        k_params_dict = {'calcite': np.array([20.1, 172.1, 24.7, 211.16, 0.0992]),'calcite_arf': np.array([21.6, 180.1, 32.5, 260.3, 0.0944]), 'calcite_ll': np.array([26.0, 204, 37.5, 290.16, 0.0903]),
        'calcite_ll_new': np.array([31.1, 237, 32.9, 262.4, 0.0914]),'dolomite_bonifacie': np.array([21.5, 194.5, 31.8, 273.6, 0.0668]),
        'dolomite_sch_first': np.array([25.3, 220.1, 31.5, 275.3, 0.0766]), 'dolomite_sch': np.array([24.2, 214.0, 31.9, 278.8, 0.0720]),
        'dolomite_arf': np.array([24.2, 214.0, 31.9, 278.8, 0.0720]),'dolomite_ll': np.array([24.3, 212.7, 32.3, 278.7, 0.0685]),
        'dolomite_ll_first': np.array([22.6, 201.7, 33.1, 283.0, 0.0734]) }

        k_sds_dict = {'calcite': np.array([0.7,5.0, 4.5, 30.0, 0.0]), 'calcite_arf': np.array([3.7,22.4, 1.0, 6.6, 0.0]),'calcite_arf_true': np.array([3.7,22.4, 1.0, 6.6, 0.0]),
        'calcite_ll': np.array([0.7,5.0, 4.5, 30.0, 0.0]), 'calcite_ll_new': np.array([6.3,37.2, 1.6, 10.3, 0.0]),
        'dolomite_bonifacie': np.array([1.08, 7.3, 2.3, 15.8, 0.0063]),'dolomite_sch_first': np.array([2.1, 13.5, 2.5, 16.3, 0.015]),
        'dolomite_sch': np.array([2.1, 13.5, 2.5, 16.3, 0.015]),'dolomite_arf': np.array([2.1, 13.5, 2.5, 16.3, 0.015]),
        'dolomite_ll': np.array([2.7, 16.8, 2.4, 15.9, 0.013]), 'dolomite_ll_first': np.array([2.1, 13.5, 2.5, 16.3, 0.015])}

        kinetic_params = k_params_dict[mineral]
        k_sds = k_sds_dict[mineral]
        pred_df, solution_details = predict_D47_s(times_yrs, time_temp_fn, D47_init, bulk_comp, kinetic_params, k_sds, sigma_choice, mineral, T_soak_ch = T_soak_choice)

    elif model == 'p':
        # schauble_dolomite included
        k_params_dict = {'calcite': np.array([19.9,187.6,21.4,180.0,14.4,136.1]),'calcite_arf': np.array([19.9,187.6,21.4,180.0,14.4,136.1]), 'dolomite': np.array([20.6,217.2,15.2,160.3,6.3,100.6]), 'dolomite_sch': np.array([20.6,217.2,15.2,160.3,6.3,100.6]) }
        k_sds_dict = {'calcite': np.array([4.4,2.7,1.0,5.7,0.5,3.7]), 'calcite_arf': np.array([4.4,2.7,1.0,5.7,0.5,3.7]),'dolomite': np.array([5.2,32.8,3.3,23.5,5.0,33.4]), 'dolomite_sch': np.array([5.2,32.8,3.3,23.5,5.0,33.4])}
        kinetic_params = k_params_dict[mineral]
        k_sds = k_sds_dict[mineral]
        pred_df, solution_details = predict_D47_p(times_yrs, time_temp_fn, D47_init, bulk_comp, kinetic_params, k_sds, sigma_choice, mineral, T_soak_ch = T_soak_choice)

    else:
        print('invalid model name')

    # apply temperature predictions now
    # because newton-raphson method is inaccurate for D47_ARF values below 0.330, need to switch between methods
    pred_df['Temp_pred'] = pred_df['D47_pred'].apply(D47_to_temp_ARF)
    pred_df['Temp_true'] = pred_df['time_yrs'].apply(time_temp_fn)
    # catch times when too fast, and error is thrown
    pred_df['Temp_pred_upper'] = pred_df['D47_pred_upper'].apply(D47_to_temp_ARF)
    # turn all D47 values lower than 0.2 into fills based on what's around them
    pred_df.loc[pred_df['D47_pred_lower']<0.20,'D47_pred_lower'] = np.nan
    pred_df['D47_pred_lower'].interpolate(method = 'linear', inplace = True)
    pred_df['Temp_pred_lower'] = pred_df['D47_pred_lower'].apply(D47_to_temp_ARF)
    newton_raphson_cutoff = 0.330
    pred_df.loc[pred_df['D47_pred'] < newton_raphson_cutoff, 'Temp_pred'] = pred_df.loc[pred_df['D47_pred'] < newton_raphson_cutoff, 'D47_pred'].apply(D47_to_temp_ARF_long)
    pred_df.loc[pred_df['D47_pred_upper'] < newton_raphson_cutoff, 'Temp_pred_upper'] = pred_df.loc[pred_df['D47_pred_upper'] < newton_raphson_cutoff, 'D47_pred_upper'].apply(D47_to_temp_ARF_long)
    pred_df.loc[pred_df['D47_pred_lower'] < newton_raphson_cutoff, 'Temp_pred_lower'] = pred_df.loc[pred_df['D47_pred_lower'] < newton_raphson_cutoff, 'D47_pred_lower'].apply(D47_to_temp_ARF_long)

    return(pred_df, solution_details)

def predict_D47_s(times_yrs, time_temp_fn, D47_init, bulk_comp, kinetic_params, k_sds, sigma_choice, mineral, T_soak_ch = False):
    # predict D47 and pair concentration using stolper model
    times_yrs = pd.Series(times_yrs)
    times_s = times_yrs.values*365.25*24*60*60
    bulks = concentration_conversion(bulk_comp[0], bulk_comp[1], D47_init)
    pairs_conc_equil = equil_pair_conc(bulks, kinetic_params[4],time_temp_fn(times_yrs))
    # actual weighted values, fix pairs to calcite slope
    k_sds_exch = np.copy(k_sds)
    k_sds_diff = np.copy(k_sds)
    #decide which errors to use
    if mineral == 'calcite':
        cov_parameter = 0.0
        k_sds_exch[2:4] = k_sds[2:4]*cov_parameter
        k_sds_exch[4] = 0.0
        k_sds_diff[0:2] = k_sds[0:2]*cov_parameter
        k_sds_diff[4] = 0.0


    else:
        cov_parameter = 0.0
        k_sds_exch[2:4] = k_sds_exch[2:4]*cov_parameter
        k_sds_exch[4] = 0.0
        k_sds_diff[0:2] = k_sds_diff[0:2]*cov_parameter
        k_sds_diff[4] = 0.0
    # extra args
    extraArgs = (kinetic_params, bulks, time_temp_fn, T_soak_ch)
    # inital y. y0[0] is rxn progress, y0[1] is concentration of pairs (inital excess)
    y0 = np.array([0, initial_pair_conc(bulks,kinetic_params[4])])
    # time, in seconds. t[0] is initial time
    t = times_s
    # insert a zero for initial params
    t = np.insert(t,0,0.0)
    # Defining ode params
#    rel_tol = 1e-14
#    abs_tol = 1e-14
    variable_tol = calculate_resolution_tolerances(times_yrs, time_temp_fn)
    rel_tol = variable_tol
    abs_tol = variable_tol
    # solve for all time points
    ode_soln = odeint(complex_model_ode, y0, t, args = extraArgs, rtol = rel_tol, atol = abs_tol, full_output = True, hmin = 1e-266, hmax = 1e66, mxstep = int(1e7))
    this_df = pd.DataFrame(np.array([times_yrs,times_s,ode_soln[0][1:,0], ode_soln[0][1:,1], pairs_conc_equil]).T,columns = ['time_yrs', 'time_s', 'rxn_progress_predicted','pairs_conc_predicted', 'pairs_conc_equil'])
    D47_pred = rxn_progress_to_D47_ARF_explicit(this_df['rxn_progress_predicted'], bulks.c63_init, bulks.c63_stoch)
    this_df['D47_pred'] = D47_pred
    pairs_rd = equil_pair_conc(bulks,kinetic_params[4],1900)
    this_df['d_pairs_pred'] = (this_df['pairs_conc_predicted']/pairs_rd-1)*1000
    this_df['d_pairs_equil'] = (this_df['pairs_conc_equil']/pairs_rd-1)*1000
    # + X sd of errors
    extraArgs_plus_exch = (kinetic_params + k_sds_exch*sigma_choice, bulks, time_temp_fn, T_soak_ch)
    ode_soln_plus_exch = odeint(complex_model_ode, y0, t, args = extraArgs_plus_exch, rtol = rel_tol, atol = abs_tol, full_output = True, hmin = 1e-266, hmax = 1e66, mxstep = int(1e7))
    extraArgs_minus_exch = (kinetic_params - k_sds_exch*sigma_choice, bulks, time_temp_fn, T_soak_ch)
    ode_soln_minus_exch = odeint(complex_model_ode, y0, t, args = extraArgs_minus_exch, rtol = rel_tol, atol = abs_tol, full_output = True, hmin = 1e-266, hmax = 1e66, mxstep = int(1e7))

    extraArgs_plus_diff = (kinetic_params + k_sds_diff *sigma_choice, bulks, time_temp_fn, T_soak_ch)
    ode_soln_plus_diff  = odeint(complex_model_ode, y0, t, args = extraArgs_plus_diff , rtol = rel_tol, atol = abs_tol, full_output = True, hmin = 1e-266, hmax = 1e66, mxstep = int(1e7))
    extraArgs_minus_diff  = (kinetic_params - k_sds_diff *sigma_choice, bulks, time_temp_fn, T_soak_ch)
    ode_soln_minus_diff  = odeint(complex_model_ode, y0, t, args = extraArgs_minus_diff, rtol = rel_tol, atol = abs_tol, full_output = True, hmin = 1e-266, hmax = 1e66, mxstep = int(1e7))

    this_df['D47_pred_upper_diff'] = rxn_progress_to_D47_ARF_explicit(ode_soln_plus_diff[0][1:,0], bulks.c63_init, bulks.c63_stoch)
    this_df['D47_pred_lower_diff'] = rxn_progress_to_D47_ARF_explicit(ode_soln_minus_diff[0][1:,0], bulks.c63_init, bulks.c63_stoch)

    this_df['D47_pred_upper_exch'] = rxn_progress_to_D47_ARF_explicit(ode_soln_plus_exch[0][1:,0], bulks.c63_init, bulks.c63_stoch)
    this_df['D47_pred_lower_exch'] = rxn_progress_to_D47_ARF_explicit(ode_soln_minus_exch[0][1:,0], bulks.c63_init, bulks.c63_stoch)

    this_df['D47_pred_upper'] = this_df.loc[:,['D47_pred_upper_diff', 'D47_pred_upper_exch', 'D47_pred_lower_diff','D47_pred_lower_exch','D47_pred']].max(axis = 1)
    this_df['D47_pred_lower'] = this_df.loc[:,['D47_pred_lower_diff', 'D47_pred_lower_exch', 'D47_pred_upper_diff', 'D47_pred_upper_exch','D47_pred']].min(axis = 1)
    this_df['pairs_pred_upper'] = pd.DataFrame(data = [ode_soln[0][1:,1], ode_soln_plus_diff[0][1:,1], ode_soln_minus_diff[0][1:,1], ode_soln_minus_exch[0][1:,1], ode_soln_plus_exch[0][1:,1]]).max(axis = 0)
    this_df['pairs_pred_lower'] = pd.DataFrame(data = [ode_soln[0][1:,1], ode_soln_plus_diff[0][1:,1], ode_soln_minus_diff[0][1:,1], ode_soln_minus_exch[0][1:,1], ode_soln_plus_exch[0][1:,1]]).min(axis = 0)

    this_df['d_pairs_pred_upper'] = (this_df['pairs_pred_upper']/pairs_rd-1)*1000
    this_df['d_pairs_pred_lower'] = (this_df['pairs_pred_lower']/pairs_rd-1)*1000
#    raise ValueError('A very specific bad thing happened.')

    return(this_df, ode_soln)
def predict_D47_p(times_yrs, time_temp_fn, D47_init, bulk_comp, kinetic_params, k_sds, sigma_choice, mineral, T_soak_ch = False):
    # predict D47 and pair concentration using passey and henkes model
    times_yrs = pd.Series(times_yrs)
    times_s = times_yrs.values*365.25*24*60*60
    Temps_C = time_temp_fn(times_yrs)
    Temps_K_equil = Temps_C + 273.15
    D47_equil = temp_to_D47_ARF(Temps_C)
    these_kinetic_params = kinetic_params[0:2]
    delta_time_s = times_s[1:] - times_s[:-1]

    # add upper and lower bounds
    kinetic_params_upper = these_kinetic_params + k_sds[0:2]*sigma_choice
    kinetic_params_lower= these_kinetic_params - k_sds[0:2]*sigma_choice
    # calcualte reaction progress for all points

    rxn_progress = henkes_first_order_eqn(Temps_K_equil, delta_time_s, these_kinetic_params)
    rxn_progress_upper = henkes_first_order_eqn(Temps_K_equil, delta_time_s, kinetic_params_upper)
    rxn_progress_lower = henkes_first_order_eqn(Temps_K_equil, delta_time_s, kinetic_params_lower)

    # Iteratively convert first-order model to predicted D47s
    D47_predicted = np.zeros(len(Temps_K_equil))
    D47_predicted_upper = np.zeros(len(Temps_K_equil))
    D47_predicted_lower = np.zeros(len(Temps_K_equil))
    #first one is starting point
    D47_predicted[0] = D47_init
    D47_predicted_upper[0] = D47_init
    D47_predicted_lower[0] = D47_init

    # iteratively solve all following points
    for i in range(1,len(D47_predicted)):
        D47_predicted[i] = np.exp(rxn_progress[i])*(D47_predicted[i-1]-D47_equil[i])+D47_equil[i]
        D47_predicted_upper[i] = np.exp(rxn_progress_upper[i])*(D47_predicted_upper[i-1]-D47_equil[i])+D47_equil[i]
        D47_predicted_lower[i] = np.exp(rxn_progress_lower[i])*(D47_predicted_lower[i-1]-D47_equil[i])+D47_equil[i]

    this_df = pd.DataFrame({'time_yrs':times_yrs, 'time_s': times_s, 'rxn_progress_predicted':rxn_progress,'D47_equil': D47_equil,
    'D47_pred': D47_predicted, 'D47_pred_upper_raw': D47_predicted_upper,'D47_pred_lower_raw': D47_predicted_lower})
    ode_soln = ['holder', {'modelType':'FirstOrder', 'message': 'Nothing else to say'}]

    this_df['D47_pred_upper'] = this_df.loc[:,['D47_pred_upper_raw', 'D47_pred_lower_raw','D47_pred']].max(axis = 1)
    this_df['D47_pred_lower'] = this_df.loc[:,['D47_pred_upper_raw', 'D47_pred_lower_raw','D47_pred']].min(axis = 1)
#    this_df['D47_pred_upper'] = this_df['D47_pred_max']
#    this_df['D47_pred_lower'] = this_df['D47_pred_min']
    return(this_df, ode_soln)

def henkes_first_order_eqn(Temps_K_equil, delta_time_s, kinetic_params):
    R = 8.314e-3
    #preallocate rxn progress
    rxn_progress =  np.zeros(len(delta_time_s)+1)
    #acutal equation,
    rxn_progress[1:] = -delta_time_s*np.exp(-kinetic_params[1]/R/Temps_K_equil[1:]+kinetic_params[0])
    return(rxn_progress)

def complex_model_ode(y0, t, *extraArgs):
    # y[0] is rxn progress, y[1] is pair concentation
    n_adjacent = 6
    kinetic_params, bulks, time_temp_fn, T_soak = extraArgs
    # get Temperature at this point
    # get time in yrs
    time_yrs = t/60/60/24/365.25
    # Temp_here = time_temp_fn(time_yrs, cooling_params, T_soak)
    Temp_here = time_temp_fn(time_yrs)

    Temp_here_K = Temp_here + 273.15
    # calculate Kd, Kf at this temp
    ks_here = calculate_kd_kf(bulks, kinetic_params[4], Temp_here)
    R = 8.314e-3
    k = np.array([0.0,0.0])
    # rate of exchange at this T
    k[0] = np.exp(-kinetic_params[1]/R/Temp_here_K + kinetic_params[0])
    k[1] = np.exp(-kinetic_params[3]/R/Temp_here_K + kinetic_params[2])

    dpsi_dt = k[0]*bulks.c60*(bulks.c63_init-y0[0])-k[0]*ks_here.k_f*y0[1]
    dpair_dt = k[0]*bulks.c60*(bulks.c63_init-y0[0])-k[0]*ks_here.k_f*y0[1] + k[1]*bulks.c61*(1-bulks.c62)**n_adjacent*bulks.c62*(1-bulks.c61)**n_adjacent-k[1]*ks_here.k_d*y0[1]
    return(np.array([dpsi_dt, dpair_dt]))

def calculate_kd_kf(bulks, empirical_slope, Temp_here):
    ''' calculates kf and kd parameters by row'''
    # Number of adjacent carbonate groups. 6 in calcite, need to check that this is true of rdolomite as well
    n_adjacent = 6
    # pair equilibrium is an average of the two possible pair concentration eqns (13a or 13b)
    D47_equil = temp_to_D47_ARF(Temp_here)
    D63_equil = D47_equil - 0.268
    c63_equil = (D63_equil/1000 + 1)*bulks.c63_stoch
    pair_equilibrium = equil_pair_conc(bulks, empirical_slope, Temp_here)
    k_f_parameter = bulks.c60*c63_equil/pair_equilibrium
    k_d_parameter = (bulks.c62*(1-bulks.c61)**n_adjacent*bulks.c61*(1-bulks.c62)**n_adjacent)/pair_equilibrium
    return(pd.Series({'k_f': k_f_parameter, 'k_d': k_d_parameter}))

def initial_pair_conc(bulks,empirical_slope):
    ''' using Stolper equation for starting pair concentration'''
    n_adjacent = 6
    c_pair_random = (bulks.c61*(1-(1-bulks.c62)**n_adjacent)+bulks.c62*(1-(1-bulks.c61)**n_adjacent))/2
    c_pair = np.exp(empirical_slope/(D47_to_temp_ARF_long(bulks.D47_init)+273.15))*c_pair_random
    return(c_pair)

def equil_pair_conc(bulks, empirical_slope, Temp_here):
    ''' using Stolper equation for starting pair concentration'''
    n_adjacent = 6
    c_pair_random = (bulks.c61*(1-(1-bulks.c62)**n_adjacent)+bulks.c62*(1-(1-bulks.c61)**n_adjacent))/2
    c_pair = np.exp(empirical_slope/(Temp_here+273.15))*c_pair_random
    return(c_pair)

def concentration_conversion(d13C, d18O, D47_init):
    ''' apply concentrations, assume bulk comps both relative to vpdb'''
    # Std ratios
    R13C_vpdb=0.011180
    R18O_vsmow=0.0020052
    R17O_vsmow=0.00038475
    d18O_vsmow = d18O*1.03092+30.92
    # Get ratios
    R13C = (d13C/1000+1)*R13C_vpdb
    R18O = (d18O_vsmow/1000+1)*R18O_vsmow
    R17O = R17O_vsmow*(R18O/R18O_vsmow)**0.528
    # Get concentrations
    c12C = 1/(1+R13C)
    c13C =R13C*c12C
    c16O = 1/(1+R18O+R17O)
    c17O = R17O*c16O
    c18O = R18O*c16O
    # Get isotopologue concentrations
    c60_stoch = c12C*(c16O**3)
    c61_stoch = c13C*(c16O**3)
    c62_stoch = 3*c12C*c18O*(c16O**2)
    c63_stoch = 3*c13C*c18O*(c16O**2)

    D63_init = D47_init - 0.2679 # for ARF
    c63_init = (D63_init/1000 + 1)*c63_stoch
    # assume that singly substituteds are stochastic. Required by the way we calculate D47
    c60, c61, c62 = (c60_stoch, c61_stoch, c62_stoch)

    return(pd.Series({'c60': c60, 'c61': c61, 'c62': c62, 'c63_stoch': c63_stoch, 'c63_init': c63_init, 'D47_init': D47_init}))

def rxn_progress_to_D47_ARF_explicit(rxn_progress_predicted, c63_init, c63_stoch):
    '''converts a rxn progess term to a D47_ARF. Assumes c63_init and c63_stoch are the same for all rows'''
    D47_predicted = ((c63_init - rxn_progress_predicted)/c63_stoch-1)*1000+ 0.268
    return(D47_predicted)

def model_wrapper_with_prompts(time_array, T_t_fn, mineral_choice = 'both', model_choice = 's', sigma_choice = 1):
    if mineral_choice == 'both':
        D47_pred_dol, solution_details_dol = predict_D47(time_array, T_t_fn, temp_to_D47_ARF(T_t_fn(time_array[0])), model = model_choice, mineral = 'dolomite_sch', sigma_choice = sigma_choice)
        D47_pred_cal, solution_details_cal = predict_D47(time_array, T_t_fn, temp_to_D47_ARF(T_t_fn(time_array[0])), model = model_choice, mineral = 'calcite_arf', sigma_choice = sigma_choice)
        D47_pred = {'dolomite': D47_pred_dol, 'calcite': D47_pred_cal}
    elif mineral_choice == 'calcite':
        D47_pred_cal, solution_details_cal = predict_D47(time_array, T_t_fn, temp_to_D47_ARF(T_t_fn(time_array[0])), model = model_choice, mineral = 'calcite_arf', sigma_choice = sigma_choice)
        D47_pred = {'calcite': D47_pred_cal}

    elif mineral_choice == 'dolomite':
        D47_pred_dol, solution_details_dol = predict_D47(time_array, T_t_fn, temp_to_D47_ARF(T_t_fn(time_array[0])), model = model_choice, mineral = 'dolomite_sch', sigma_choice = sigma_choice)
        D47_pred = {'dolomite': D47_pred_dol}
    return(D47_pred)

def monte_carlo_model_looper(T_t_points, model_iterations, mineral_choice = 'both', sigma_choice = 1):
    rd = random.SystemRandom()
    model_resoluion = 1e5
    time_array = np.linspace(0, T_t_points['time_yrs'].max(), num = model_resoluion)
    if mineral_choice == 'both':
        dol_pred = []
        cal_pred = []
        for i in range(model_iterations):
            print('Model numer {0}...'.format(i))
            this_T_t_fn = monte_carlo_T_t_path(T_t_points)
            this_D47_pred_dol, these_solution_details_dol = predict_D47(time_array, this_T_t_fn, temp_to_D47_ARF(this_T_t_fn(time_array[0])), model = 's', mineral = 'dolomite_sch', sigma_choice = sigma_choice)
            this_D47_pred_cal, these_solution_details_cal = predict_D47(time_array, this_T_t_fn, temp_to_D47_ARF(this_T_t_fn(time_array[0])), model = 's', mineral = 'calcite', sigma_choice = sigma_choice)
            dol_pred.append(this_D47_pred_dol)
            cal_pred.append(this_D47_pred_cal)
        D47_pred_MC = {'dolomite': dol_pred, 'calcite': cal_pred}
    elif mineral_choice == 'calcite':
        cal_pred = []
        for i in range(model_iterations):
            print('Model numer {0}...'.format(i))
            this_T_t_fn = monte_carlo_T_t_path(T_t_points)
            this_D47_pred_cal, these_solution_details_cal = predict_D47(time_array, this_T_t_fn, temp_to_D47_ARF(this_T_t_fn(time_array[0])), model = 's', mineral = 'calcite', sigma_choice = sigma_choice)
            cal_pred.append(this_D47_pred_cal)
        D47_pred_MC = {'calcite': cal_pred}
    elif mineral_choice == 'dolomite':
        dol_pred = []
        for i in range(model_iterations):
            print('Model numer {0}...'.format(i))
            this_T_t_fn = monte_carlo_T_t_path(T_t_points)
            this_D47_pred_dol, these_solution_details_dol = predict_D47(time_array, this_T_t_fn, temp_to_D47_ARF(this_T_t_fn(time_array[0])), model = 's', mineral = 'dolomite_sch', sigma_choice = sigma_choice)
            dol_pred.append(this_D47_pred_dol)
        D47_pred_MC = {'dolomite': dol_pred}
    return(D47_pred_MC)

def monte_carlo_T_t_path(T_t_points, rd_engine = random.SystemRandom()):
    # generate random T-t path based on boxes
    for line, row in T_t_points.iterrows():
        row['time_rd'] = rd_engine.uniform(row['time_lower'], row['time_upper'])
        row['temp_rd'] = rd_engine.uniform(row['temp_lower'], row['temp_upper'])
    # fit T-t points to get fn
    T_t_fn = PchipInterpolator(T_t_points['time_rd'], T_t_points['temp_rd'])
    # return this fn
    return(T_t_fn)

def add_box_columns(T_t_points):
    T_t_points['time_rd'] = T_t_points['time_yrs']
    T_t_points['temp_rd'] = T_t_points['temp_C']
    # get lower time limits here
    T_t_points['time_lower'] = T_t_points['time_yrs'] - T_t_points['time_width_yrs']/2.0
    # round up so that no negative times
    T_t_points.loc[T_t_points['time_lower'] < 0, 'time_lower'] = 0.0
    # get upper time limits here
    T_t_points['time_upper'] = T_t_points['time_yrs'] + T_t_points['time_width_yrs']/2.0
    # round down so that no times beyond predefined max time
    T_t_points.loc[T_t_points['time_upper'] > T_t_points['time_yrs'].max(), 'time_upper'] = T_t_points['time_yrs'].max()
    # Temp upper and lower
    T_t_points['temp_lower'] = T_t_points['temp_C'] - T_t_points['temp_width_C']/2.0
    T_t_points['temp_upper'] = T_t_points['temp_C'] + T_t_points['temp_width_C']/2.0

    return(T_t_points)
