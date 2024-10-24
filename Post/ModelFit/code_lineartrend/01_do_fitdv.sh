#!/bin/bash

for st in `cat trim.slst`
do
  fn=../../PREP_data/INTERP_${st}.csv
  bt=`gawk -F, 'NR==2{print $9}' $fn`
  et=`gawk -F, 'END{print $9}' $fn`

  echo python modelfit_MCMC_Utah-soil.py       $st  $bt $et 
  echo python modelfit_MCMC_Utah-soil-temp.py  $st  $bt $et
  echo python modelfit_MCMC_Utah-temp.py       $st  $bt $et
done
