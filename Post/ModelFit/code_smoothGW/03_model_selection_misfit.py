import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

stnm=sys.argv[1]
modelcase=["soil_lake","temp_lake","soil_temp_lake"] #,"snow","soil_snow","temp_snow","soil_temp_snow"]
#nump=[5,6,7]
nump=[4,5,6]

n=0
opaic=np.array(range(len(modelcase)))
opbic=np.array(range(len(modelcase)))
opstn=[]
fname="Misfit_"+stnm.strip()+".csv"
f=open(fname,'w')
# print("stan,modelcase,aic,bic,numpara")
f.write("stan,modelcase,aic,bic,numpara,norm_misfit,misfit\n")
plt.figure(figsize=(4,4))
for mod in modelcase:
    uu=pd.read_csv('./BestFit_csv/Fit_'+stnm+'_'+mod+'.csv', sep=',')
    df_uu = pd.DataFrame(uu)

    x=np.array(df_uu.dv)
    y=np.array(df_uu.bestfit) 
    err=np.array(df_uu.err)
    # Calculate the log-likelihood of the model
    # log_likelihood =  np.sum((x - y)**2)/len(x)

    misfit =  np.sum((x - y)**2)/len(x)
    norm_misfit = 0

    log_likelihood = 0
    for i in range(len(x)):
        norm_misfit += ((x[i] - y[i])**2/err[i]**2)
        log_likelihood += ((x[i] - y[i])**2/err[i]**2)
    log_likelihood=log_likelihood/len(x)
    norm_misfit = norm_misfit/len(x)

    # Calculate the number of parameters in the model
    num_parameters = nump[n]
    # Calculate the AIC
    aic =  (len(x)/60) * np.log(log_likelihood) + 2 * num_parameters
    
    # Calculate the BIC
    bn = len(y)  # Number of data points
    bic = (len(x)/60) * np.log(log_likelihood) + num_parameters * np.log(bn)

    line=stnm+","+mod+",%.2f"%aic+",%.2f"%bic+",%d"%num_parameters+",%4f"%norm_misfit+",%4f"%misfit
    # print(line)
    n=n+1
    f.write(line+"\n")

#     # plt.plot(aic,bic,label=str(mod),marker='o', alpha=0.5)

# # print(stnm, " data num:", len(x), len(x)/60)
    
# figname='AB_'+stnm+'_fitdv.png'
# plt.title(stnm)
# plt.xlabel("AIC value")
# plt.ylabel("BIC value")
# plt.tight_layout()
# plt.legend()
# plt.savefig(figname)

f.close()
