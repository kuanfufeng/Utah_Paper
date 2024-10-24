import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import zip_longest

#import scipy.interpolate
from scipy.interpolate import make_interp_spline, interp1d, CubicSpline, PchipInterpolator


if __name__ == '__main__':

    #-------------------------------------------#
    output_imgdir = "figure"
    if not os.path.exists(output_imgdir):
        os.makedirs(output_imgdir)
    #------------------------------------------------------#
    
    stnm=sys.argv[1] #'SPU'
    root="../UU_csv_blank/"
    utvec=np.loadtxt(root+"time.lst",dtype=float)
    
    
    # --- Read dv and factors 

    cols1=['time', 'dv', 'err','date']
    fn=root+'DV_'+stnm+'.csv'
    fiv=pd.read_csv(fn,names=cols1,header=0)
    t2 = np.array(fiv["time"])
    dvv = np.array(fiv["dv"])
    err = np.array(fiv["err"])
    vdate=np.array(fiv["date"])

    
    cols1=['ftime',  'temp_prism', 'ppt_prism', 'soil_nldas', 'snow_nldas', 'date']
    fn=root+'pretrim_'+stnm+'.csv'
    fi=pd.read_csv(fn,names=cols1,header=0)
    temp = ((np.array(fi["temp_prism"])) -np.mean(fi["temp_prism"]))   #/np.max(np.abs(fi["temp"]-np.mean(fi["temp"])))
    soil = ((np.array(fi["soil_nldas"])) -np.mean(fi["soil_nldas"]))   #/np.max(np.abs(fi["soil"]-np.mean(fi["soil"])))
    snow = (np.array(fi["snow_nldas"]) -np.mean(fi["snow_nldas"]))   
    ftime=np.array(fi["ftime"])

    #---Read Lake data---# 
    cols2=["utime","GSL","UL"]
    root2="/home/kffeng/WORKs/Utah_dv_Paper/UU_hydro/"
    fn2= root2+"Lakes.csv"
    fi2 = pd.read_csv(fn2,names=cols2,header=0)
    
    utvec2=np.array(fi2["utime"])
    GSL = (np.array(fi2["GSL"]) -np.mean(fi2["GSL"]))  #/np.max(np.abs(fi2["GSL"]-np.mean(fi2["GSL"])))
    UL  = (np.array(fi2["UL"])  -np.mean(fi2["UL"]))   #/np.max(np.abs(fi2["UL"]-np.mean(fi2["UL"])))
    
    #gwf=root2+"DUG_well.tmp"
    #gwt,dep2gw =np.loadtxt(gwf,delimiter=' ',unpack=True)
    #DEP2GW=(dep2gw-np.mean(dep2gw))/np.max(np.abs(dep2gw-np.mean(dep2gw)))
    
    #------------------------------------------------------#
    intdv = np.interp(utvec, t2, dvv)
    fint = interp1d(utvec, intdv)
    fspl = interp1d(utvec, intdv, kind='cubic')
    fcs = CubicSpline(t2, dvv)
    fcsp = PchipInterpolator(t2, dvv)
    fcsm = make_interp_spline(t2, dvv)
    cs_err = CubicSpline(t2, err)
    
    intGSL = np.interp(utvec, utvec2,GSL)
    intUL =  np.interp(utvec, utvec2,UL)
    
    cs_gsl=CubicSpline(utvec2, GSL)
    cs_ul=CubicSpline(utvec2, UL)
    cs_temp=CubicSpline(ftime,temp)
    cs_soil=CubicSpline(ftime,soil)
    cs_snow=CubicSpline(ftime,snow)
    
    #intgw=np.interp(utvec, gwt,DEP2GW)
    #gwcs=CubicSpline(gwt,DEP2GW)
    
    
    # --- define begin and end time
    tb=t2[0]
    te=t2[-1]
    # --- limited by soil data
    if (te>=2022.6250): # from soil
        for nt in utvec:
            if (nt>=2022.6250):    # from Lakes
                te=nt
                break
    
    Ntb=np.where(utvec==tb)[0][0]
    Nte=np.where(utvec==te)[0][0]
    print(" # --- ",stnm," Data period: ",tb,te,Nte-Ntb)
    
    vdate2=pd.date_range(vdate[0],vdate[-1],freq='D')
    data_date=pd.date_range(vdate2[0],vdate2[Nte-Ntb-1],freq='D')
    print("data range: ",data_date[0],data_date[-1])
    
    # --- define data
    data_time=utvec[Ntb:Nte]
    data_dv=fcsp(utvec)[Ntb:Nte]
    data_err=cs_err(utvec)[Ntb:Nte]
    data_temp=cs_temp(utvec)[Ntb:Nte]
    data_soil=cs_soil(utvec)[Ntb:Nte]
    data_snow=cs_snow(utvec)[Ntb:Nte]
    #data_gw=gwcs(utvec)[Ntb:Nte]
    data_GSL=cs_gsl(utvec)[Ntb:Nte]
    data_UL=cs_ul(utvec)[Ntb:Nte]

    fieldnames = ['utvec', 'dv','err','temp','SM_EWT','snow_EWT','GSL','UL','date']
    fcsv="INTERP_"+stnm+".csv"
    data={
        'utvec': data_time,
        'dv': data_dv,
        'err': data_err,
        'temp':data_temp,
        'SM_EWT':data_soil,
        'snow_EWT':data_snow,
        'GSL':data_GSL,
        'UL':data_UL,
        'date':data_date    
    }
    
    df=pd.DataFrame(data)
    df.to_csv(fcsv,columns=fieldnames,sep=',',index = None, header=True, float_format="%.6f" )
    
    print("plotting ")
    #---plot dv/v for the debug---#
    fig, ax = plt.subplots(6, 1, figsize=(6,12))
    
    for k in range(0,6):
        ax[k].set_xlim(data_time[0],data_time[-1])
        #ax[k].set_ylim(-1,1)
    
    #ax[0].set_ylim(-1,1)
    
    ax[0].plot(t2,dvv,'-',c='green',linewidth=10,alpha=0.5)
    #ax[0].plot(data_time,data_dv,'-',c='m')
    ax[0].plot(data_time,fcs(utvec)[Ntb:Nte],'-',c='m')
    ax[0].plot(data_time,fcsp(utvec)[Ntb:Nte],c='b',linewidth=5,alpha=0.5)
    ax[0].plot(data_time,fcsm(utvec)[Ntb:Nte],'--',c='cyan',linewidth=1)
    ax[0].set_title(stnm)
    
    ax[1].plot(ftime,temp,  ls="-", c = "orange",linewidth=5,alpha=0.5)
    ax[1].plot(data_time,data_temp,  ls="-", c = "r")
    ax[1].set_title(" temp")
    
    ax[2].plot(ftime,soil,  ls="-", c = "blue",linewidth=5,alpha=0.5)
    ax[2].plot(data_time,data_soil,  ls="-", c = "cyan")
    ax[2].set_title(" soil EWT (m)")
    
    ax[3].plot(ftime,snow,  ls="-", c = "blue",linewidth=5,alpha=0.5)
    ax[3].plot(data_time,data_snow,  ls="-", c = "cyan")
    ax[3].set_title(" snow EWT (m)")
    
    ax[4].plot(utvec,intGSL,  ls="-", c = "b",linewidth=5,alpha=0.5)
    ax[4].plot(data_time,data_GSL,  ls="-", c = "k")
    ax[4].set_title(" Great Salt Lake water level")
    
    ax[5].plot(utvec,intUL,  ls="-", c = "lime",linewidth=5,alpha=0.5)
    ax[5].plot(data_time,data_UL,  ls="-", c = "b")
    ax[5].set_title(" Utah Lake water level")
    '''
    ax[5].plot(utvec,intgw,  ls="-", c = "grey",linewidth=5,alpha=0.5)
    ax[5].plot(gwt,DEP2GW,  "o", c = "k")
    ax[5].plot(data_time,data_gw,  ls="-", c = "b")
    ax[5].set_title(" groundwater level (DUG)")
    '''
    plt.tight_layout()
    print("save")
    plt.savefig(output_imgdir+"/TSdvv_%s.png"%(stnm), format="png", dpi=100)
    plt.close()
    print("close")
    plt.clf()