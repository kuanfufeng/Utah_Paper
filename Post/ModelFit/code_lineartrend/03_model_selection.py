import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

stnm=sys.argv[1]
modelcase=["soil","temp","soil_temp"] #,"snow","soil_snow","temp_snow","soil_temp_snow"]
#nump=[4,5,6]
nump=[3,4,5]

n=0
opaic=np.array(range(len(modelcase)))
opbic=np.array(range(len(modelcase)))
opstn=[]
fname="AICBIC_"+stnm.strip()+".csv"
f=open(fname,'w')
print("stan,modelcase,aic,bic,numpara")
f.write("stan,modelcase,aic,bic,numpara\n")
plt.figure(figsize=(4,4))
for mod in modelcase:
    uu=pd.read_csv('./BestFit_csv/Fit_'+stnm+'_'+mod+'.csv', sep=',')
    df_uu = pd.DataFrame(uu)

    x=np.array(df_uu.dv)
    y=np.array(df_uu.bestfit) 
    # Calculate the log-likelihood of the model
    log_likelihood =  np.sum((x - y)**2)/len(x)
    # Calculate the number of parameters in the model
    num_parameters = nump[n]
    # Calculate the AIC
    aic =  len(x) * np.log(log_likelihood) + 2 * num_parameters
    # Calculate the BIC
    bn = len(y)  # Number of data points
    bic = len(x) * np.log(log_likelihood) + num_parameters * np.log(bn)

    line=stnm+","+mod+",%.2f"%aic+",%.2f"%bic+",%d"%num_parameters
    print(line)
    n=n+1
    f.write(line+"\n")

    plt.plot(aic,bic,label=str(mod),marker='o', alpha=0.5)

figname='AB_'+stnm+'_fitdv.png'
plt.title(stnm)
plt.xlabel("AIC value")
plt.ylabel("BIC value")
plt.tight_layout()
plt.legend()
plt.savefig(figname)

f.close()
