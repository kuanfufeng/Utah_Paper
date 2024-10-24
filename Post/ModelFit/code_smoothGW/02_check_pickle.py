import pickle
import emcee
import pandas as pd
import numpy as np
import pprint
import corner
import matplotlib.pyplot as plt
import sys
# import functions for MCMC
from MCMC_func import *
from utils import *


def sample_walkers_soil_temp(nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model_soil_temp_lake(i, all=False, **modelparam)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread
#-------------------------------------------#
def sample_walkers_soil(nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model_soil_lake(i, all=False, **modelparam)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread
#-------------------------------------------#
def sample_walkers_temp(nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model_temp_lake(i, all=False, **modelparam)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread
#-------------------------------------------#

freq="2-4"

nsteps=12000
burnin=1200
thin=120
fnst='trim.slst'

#-------------------------------------------#
model="soil_temp_lake"
labels = ["A","B","C","t_{shiftdays}","D","w_{shiftdays}", "log(f)"]

ftxt="ModelCase_"+model+"_scalar_coef.txt"
file = open(ftxt, "w")
content="Fitting_results,stnm,A,B,C,tshift,D,wshift,logf,nsteps,nwlker\n"
print("output content to scalar_coef.txt ---  ",content.strip())
file.write(content)
with open(fnst, 'r') as f:
    for stnm in f:
        pair=stnm.strip()+"-"+stnm.strip()
        pkfn="../processed_fitdv_"+model+"/MCMC_sampler_"+str(nsteps)+"_Normalized/MCMC_sampler_"+pair+"_"+freq+"_"+model+"_stretching.pickle"
        print(pkfn,"\n")
        
        with open(pkfn,'rb') as f:
            sampler=pickle.load(f)
            modelparam=pickle.load(f)

            samples = sampler.flatchain
            theta_max  = samples[np.argmax(sampler.flatlnprobability)]
            ndim=modelparam["ndim"]
            model=modelparam["modelcase"]
            unix_tvec = modelparam["unix_tvec"]
            uniform_tvec=modelparam["uniform_tvec"]
            dvv_data=modelparam["dvv_data_trim"]

            med_model, spread = sample_walkers_soil_temp(nsteps,samples)
            best_fit_model = model_soil_temp_lake(theta_max, all=False, **modelparam)
            res = dvv_data - best_fit_model
        #-------------------------------------------#
            # --- print results to file
            temp=[" %.6f " % k for k in theta_max]
            coef=[float(string) for string in temp]
            content="Fitting_results,"+stnm.strip()+",%f,%f,%f,%f,%f,%f,%f"%(coef[0],coef[1],coef[2],coef[3],coef[4],coef[5],coef[6])+","+str(nsteps)+","+str(modelparam["nwalkers"])+"\n"
            print("output content to scalar_coef.txt ---  ",content.strip())
            file.write(content)
        fieldnames = ['utvec', 'dv','bestfit','Resi','sdv']
        fcsv="Fit_"+stnm.strip()+"_"+model+".csv"
        data={
        'utvec': uniform_tvec,
        'dv':  dvv_data,
        'bestfit': best_fit_model,
        'Resi': res,
        'sdv': spread
        }
        df=pd.DataFrame(data)
        df.to_csv(fcsv,columns=fieldnames,sep=',',index = None, header=True, float_format="%.6f" )
    file.close()
#-------------------------------------------#

#-------------------------------------------#
model="soil_lake"
labels = ["A","B","D","w_{shiftdays}", "log(f)"]

ftxt="ModelCase_"+model+"_scalar_coef.txt"
file = open(ftxt, "w")
content="Fitting_results,stnm,A,B,D,wshift,logf,nsteps,nwlker\n"
print("output content to scalar_coef.txt ---  ",content.strip())
file.write(content)
with open(fnst, 'r') as f:
    for stnm in f:
        pair=stnm.strip()+"-"+stnm.strip()
        pkfn="../processed_fitdv_"+model+"/MCMC_sampler_"+str(nsteps)+"_Normalized/MCMC_sampler_"+pair+"_"+freq+"_"+model+"_stretching.pickle"
        print(pkfn,"\n")
        
        with open(pkfn,'rb') as f:
            sampler=pickle.load(f)
            modelparam=pickle.load(f)

            samples = sampler.flatchain
            theta_max  = samples[np.argmax(sampler.flatlnprobability)]
            ndim=modelparam["ndim"]
            model=modelparam["modelcase"]
            unix_tvec = modelparam["unix_tvec"]
            uniform_tvec=modelparam["uniform_tvec"]
            dvv_data=modelparam["dvv_data_trim"]

            med_model, spread = sample_walkers_soil(nsteps,samples)
            best_fit_model = model_soil_lake(theta_max, all=False, **modelparam)
            res = dvv_data - best_fit_model
        #-------------------------------------------#
            # --- print results to file
            temp=[" %.6f " % k for k in theta_max]
            coef=[float(string) for string in temp]
            content="Fitting_results,"+stnm.strip()+",%f,%f,%f,%f,%f"%(coef[0],coef[1],coef[2],coef[3],coef[4])+","+str(nsteps)+","+str(modelparam["nwalkers"])+"\n"
            print("output content to scalar_coef.txt ---  ",content.strip())
            file.write(content)
        fieldnames = ['utvec', 'dv','bestfit','Resi','sdv']
        fcsv="Fit_"+stnm.strip()+"_"+model+".csv"
        data={
        'utvec': uniform_tvec,
        'dv':  dvv_data,
        'bestfit': best_fit_model,
        'Resi': res,
        'sdv': spread 
        }
        df=pd.DataFrame(data)
        df.to_csv(fcsv,columns=fieldnames,sep=',',index = None, header=True, float_format="%.6f" )
    file.close()

#-------------------------------------------#
model="temp_lake"
labels = ["A","C","t_{shiftdays}","D","w_{shiftdays}", "log(f)"]

ftxt="ModelCase_"+model+"_scalar_coef.txt"
file = open(ftxt, "w")
content="Fitting_results,stnm,A,C,tshift,D,wshift,logf,nsteps,nwlker\n"
print("output content to scalar_coef.txt ---  ",content.strip())
file.write(content)
with open(fnst, 'r') as f:
    for stnm in f:
        pair=stnm.strip()+"-"+stnm.strip()
        pkfn="../processed_fitdv_"+model+"/MCMC_sampler_"+str(nsteps)+"_Normalized/MCMC_sampler_"+pair+"_"+freq+"_"+model+"_stretching.pickle"
        print(pkfn,"\n")
        
        with open(pkfn,'rb') as f:
            sampler=pickle.load(f)
            modelparam=pickle.load(f)

            samples = sampler.flatchain
            theta_max  = samples[np.argmax(sampler.flatlnprobability)]
            ndim=modelparam["ndim"]
            model=modelparam["modelcase"]
            unix_tvec = modelparam["unix_tvec"]
            uniform_tvec=modelparam["uniform_tvec"]
            dvv_data=modelparam["dvv_data_trim"]

            med_model, spread = sample_walkers_temp(nsteps,samples)
            best_fit_model = model_temp_lake(theta_max, all=False, **modelparam)
            res = dvv_data - best_fit_model
        #-------------------------------------------#
            # --- print results to file
            temp=[" %.6f " % k for k in theta_max]
            coef=[float(string) for string in temp]
            content="Fitting_results,"+stnm.strip()+",%f,%f,%f,%f,%f,%f"%(coef[0],coef[1],coef[2],coef[3],coef[4],coef[5])+","+str(nsteps)+","+str(modelparam["nwalkers"])+"\n"
            print("output content to scalar_coef.txt ---  ",content.strip())
            file.write(content)
        fieldnames = ['utvec', 'dv','bestfit','Resi','sdv']
        fcsv="Fit_"+stnm.strip()+"_"+model+".csv"
        data={
        'utvec': uniform_tvec,
        'dv':  dvv_data,
        'bestfit': best_fit_model,
        'Resi': res,
        'sdv': spread 
        }
        df=pd.DataFrame(data)
        df.to_csv(fcsv,columns=fieldnames,sep=',',index = None, header=True, float_format="%.6f" )
    file.close()
#-------------------------------------------# 

