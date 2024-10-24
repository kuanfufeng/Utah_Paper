import numpy as np
import pandas as pd
import matplotlib as plt
import sys

usecols=["utvec", "dv", "err", "temp", "soil", "snow", "GSL", "UL", "date"]
root = "../PREP_data/"

fdv="CC-dv.csv"
file_dv = open(fdv, "w")
content="stan,cc(dv,dv),cc(dv,temp),cc(dv,GSL),cc(dv,UL),cc(dv,soil),cc(dv,snow)\n"
file_dv.write(content)


for st in np.loadtxt('trim.slst', dtype=str):
    stnm=st.strip()
    fn = root+"INTERP_"+stnm+".csv"
    fi = pd.read_csv(fn,names=usecols,header=0)
    fi['date'] = fi['date'].astype(str)
    dtime=np.array(fi.utvec)
    dvv  =(np.array(fi.dv)  -np.mean(fi.dv)  )
    temp =(np.array(fi.temp)-np.mean(fi.temp))
    soil =(np.array(fi.soil)-np.mean(fi.soil))
    snow =(np.array(fi.snow)-np.mean(fi.snow))
    GSL  =(np.array(fi.GSL) -np.mean(fi.GSL) )
    UL   =(np.array(fi.UL)  -np.mean(fi.UL)  )

    dvv  =dvv /np.max(np.abs(dvv))
    temp =temp  /np.max(np.abs(temp ))
    soil =soil  /np.max(np.abs(soil ))
    snow =snow  /np.max(np.abs(snow ))
    GSL  =GSL   /np.max(np.abs(GSL  ))
    UL   =UL    /np.max(np.abs(UL   ))

    cc_dv=np.ndarray([6])
    cc_dv[0]=np.corrcoef(dvv,dvv)[0,1] 
    cc_dv[1]=np.corrcoef(dvv,temp)[0,1]
    cc_dv[2]=np.corrcoef(dvv,GSL)[0,1]
    cc_dv[3]=np.corrcoef(dvv,UL)[0,1]
    cc_dv[4]=np.corrcoef(dvv,soil)[0,1]
    cc_dv[5]=np.corrcoef(dvv,snow)[0,1]

    content="%s,"%stnm+"%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"%(cc_dv[0],cc_dv[1],cc_dv[2],cc_dv[3],cc_dv[4],cc_dv[5])+"\n"
    #print(content.strip())
    file_dv.write(content)
    
    