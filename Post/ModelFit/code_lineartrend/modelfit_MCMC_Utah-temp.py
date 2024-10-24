#!/usr/bin/env python
# coding: utf-8

# # MCMC model fitting
# 2021.11.24 Kurama Okubo
#
# 2022.1.18 update to speed up iteration and clean up the notebook.
#
# 2022.10.5 update AIC and model selection

# This notebook conduct MCMC model fitting to estimate model parameters as well as showning the multimodality.

import datetime
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "16"

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import h5py
import pickle

from tqdm import tqdm


# For the speed up of integral with Low level calling functoin
import ctypes
from scipy import LowLevelCallable

import emcee # MCMC sampler
import corner

# import functions for MCMC
from MCMC_func import *

from multiprocessing import Pool, cpu_count

#plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
os.environ['TZ'] = 'GMT' # change time zone to avoid confusion in unix_tvec conversion

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

def sample_walkers(nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model_temp(i, all=False, **modelparam)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread


if __name__ == '__main__':
    
    stnm=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    # fitting start and end time
    starttime = datetime.datetime(int(sdate[0:4]), int(sdate[5:7]), int(sdate[8:10]))
    endtime = datetime.datetime(int(edate[0:4]), int(edate[5:7]), int(edate[8:10]))

    # parameters for stacking; identical to the input file of the stacking
    cc_time_unit=86400 # short-stacking time unit
    averagestack_factor=1 # length of time bin to compute mean and std
    averagestack_step=1
    datefreq = '%dD'%(averagestack_step*cc_time_unit/86400)
    

    # select the number of iteration during the MCMC inversion
    nsteps = 12000 #30000
    burnin=1200
    thin=120
    
    # set initial value and boundaries of the model parameters
    # format is: (initial value, [vmin, vmax])
    # offset, scale of GWL, delay in GWL, scale of Temp, shift of temp in days, scale in coseimic change, healing duration for SS and PF and linear trend.
    modelparam = {
                "A"            : (0.0, [-1.0, 1.0]), # offset
                "C"            : (0.1, [0, np.inf]), # scale of Temp
                "t_{shiftdays}"   : (0, [0, 90]), # shift of temp in days
                "b_{lin}"         : (0.0, [0, np.inf]), # slope of linear trend
                "logf"         : (0.0, [-10, 10]), # magnitude of uncertainity
                }

    # model case
    modelparam["modelcase"] = "temp" 

    # MCMC parameters
    modelparam["nwalkers"] =  32 # number of chains

    #output_imgdir = "../figure/MCMC_modelfit"
    output_imgdir = "../figure/Fitdv/MCMC_temp"
    output_datadir = "../processed_fitdv_temp/MCMC_sampler_{}_Normalized".format(nsteps)

    # set random seed to fix the initial chains
    np.random.seed(seed=20121115)
    #-------------------------------------------#
    if not os.path.exists(output_imgdir):
        os.makedirs(output_imgdir)

    if not os.path.exists(output_datadir):
        os.makedirs(output_datadir)

    #---Read keys from filename---# 
    freqband = "2-4"
    dvvmethod = "stretching"

    #---Read csv containing channel-weighted dvv and err---#
    #fi = h5py.File(h5_stats_list[h5_id], "r")
    usecols=["utvec", "dv", "err", "temp", "soil", "snow", "GSL", "UL", "date"]
    root = "../../PREP_data_demean/"
    fn = root+"INTERP_"+stnm+".csv"
    fi = pd.read_csv(fn,names=usecols,header=0)
    fi['date'] = fi['date'].astype(str)

    #---Compute unix time vector---#
    tolen=len(fi['date'][:])
    btimestamp=time.mktime(time.strptime(str(fi['date'][0]), "%Y-%m-%d"))
    
    #uniform_tvec_unix = np.array(fi['uniform_tvec'])
    uniform_tvec_unix = np.array( [(btimestamp+ 86400*x) for x in range(0,tolen)]) 
    uniform_tvec = np.array([datetime.datetime.utcfromtimestamp(x) for x in uniform_tvec_unix])
    unix_tvec = np.array([x.timestamp() for x in uniform_tvec])

    modelparam["averagestack_step"] = averagestack_step
    modelparam["uniform_tvec"] = uniform_tvec
    modelparam["unix_tvec"] = unix_tvec
    
    #---Trim the time series from the starttime to the endtime---#
    fitting_period_ind = np.where((uniform_tvec >= starttime) & (uniform_tvec <= endtime))[0]
    modelparam["fitting_period_ind"] = fitting_period_ind
    print('fitting_period_ind ',fitting_period_ind)

    #---Read temperature and precipitation data at Parkfield---# 
    # Synchronize the long-period temperature and precipitation
    # store time history of trimmed precipitation and temperature
    
    modelparam["CAVG"]=(np.array(fi['temp'])-np.mean(np.array(fi['temp'])))
    modelparam["CAVG"]=modelparam["CAVG"]/np.max(np.abs(modelparam["CAVG"]))
    #---Generate the initial model parameters with chains---#

    pos, ndim, keys = get_init_param(**modelparam)

    modelparam["pos"] = pos
    modelparam["ndim"] = ndim
    modelparam["keys"] = keys

    #---Extract station pairs used for model fitting---#
    stationpairs = [stnm+'-'+stnm]
    print(stationpairs)

    #------------------------------------------------------#

    # select station ids for debug
    stationids = range(len(stationpairs))

    for stationpair in tqdm([stationpairs[i] for i in stationids]):

        print("start processing :"+stationpair)

        # search file and skip if it exists.
        foname = output_datadir+"/MCMC_sampler_%s_%s_%s_%s.pickle"%(stationpair, freqband, modelparam["modelcase"], dvvmethod)

        if os.path.exists(foname):
            print(os.path.basename(foname) + "exists. Skipping.")
            # continue;
        dvv_data = np.array(fi["dv"])
        err_data = np.array(fi["err"])

        #---plot dv/v for the debug---#
        fig, ax = plt.subplots(3, 1, figsize=(8,6))
        ax[0].errorbar(uniform_tvec, dvv_data, yerr = err_data, capsize=3, ls="-", c = "r", ecolor='black')
        ax[0].set_title(stationpair)
        ax[1].plot(uniform_tvec,modelparam["CAVG"],  ls="-", c = "orange")
        ax[1].set_title("temperature degree C from PRISM")
        plt.tight_layout()
        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.grid(True)
        plt.close()
        plt.clf()

        #---Trim the dvv and err time history---#
        modelparam["dvv_data_trim"] =  dvv_data #[fitting_period_ind]
        modelparam["err_data_trim"] =  err_data #[fitting_period_ind]
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                            modelparam["nwalkers"], modelparam["ndim"], log_probability,
                            # moves=[(emcee.moves.StretchMove(), 0.5),
                            #          (emcee.moves.DESnookerMove(), 0.5),], # Reference of Move: https://github.com/lermert/cdmx_dvv/blob/main/m.py
                            moves=emcee.moves.StretchMove(),
                            kwargs=(modelparam),
                            pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            
        # Save the current state.
        with open(foname, "wb") as f:
            pickle.dump(sampler, f)
            pickle.dump(modelparam, f)
 

        
        labels = ["A","C","t_{shiftdays}","b_{lin}", "log(f)"]

        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        
        print("theta_max:  ",theta_max)
        print(samples.shape)

        title=[" %.2f " % k for k in theta_max]
        print("Title, ",title)

        
        ### --- plotting corner
        #flat_samples = sampler.flatchain
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        print(flat_samples.shape)
        fig = corner.corner( flat_samples, show_titles=True, title_fmt=".1f", labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=theta_max,title_kwargs={"fontsize": 10})
        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s_corner.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()

        med_model, spread = sample_walkers(nsteps,samples)
        best_fit_model = model_temp(theta_max, all=False, **modelparam)

        fig, ax = plt.subplots(3, 1, figsize=(12,10))
        for theta in samples[np.random.randint(len(samples),size=9000)]:
            ax[0].plot(uniform_tvec, model_temp(theta, all=False, **modelparam), color="r", alpha=0.1)
        ax[0].plot(uniform_tvec,best_fit_model, c='b',linewidth=0.8, label='Highest Likelihood Model')
        ax[0].plot(uniform_tvec, dvv_data, label='Observed dvv', c='k')
        ax[0].set_title(stnm+" --> theta_max: "+str(title))
        ax[1].plot(uniform_tvec, dvv_data, label='observed dvv',c='k')
        ax[1].fill_between(uniform_tvec,med_model-spread*2,med_model+spread*2,color='pink',alpha=0.3,label=r'$2\sigma$ Posterior Spread')
        ax[1].fill_between(uniform_tvec,med_model-spread,med_model+spread,color='gold',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
        ax[1].plot(uniform_tvec, med_model, c='orange',linewidth=2, label='Median Model')
        ax[1].plot(uniform_tvec, best_fit_model, c='b',linestyle='--',linewidth=0.8, label='Highest Likelihood Model')

        ax[2].plot(uniform_tvec, dvv_data, label='observed dvv',c='k')
        ax[2].plot(uniform_tvec, best_fit_model, c='b',linestyle='--', label='Highest Likelihood Model')
        ax[2].plot(uniform_tvec, dvv_data-best_fit_model, label='Residual',c='r')
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(output_imgdir+"/Samp_MCMCdvv_%s_%s_%s_samplers.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()        

                
        fig, axes = plt.subplots( modelparam["ndim"], figsize=(10, 7), sharex=True)
        samples = sampler.get_chain(discard=burnin, thin=thin)
        print("samples plot : ", samples.shape, len(samples),nsteps)
        
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s_steps.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()
    
        
               


