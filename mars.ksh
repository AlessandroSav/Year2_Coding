## script to retrieve ERA5 tendencies from MARS dataset
# ParamID: 235005 to 235008
# Mean temperature tendency due to parametrisations
# Mean specific humidity tendency due to parametrisations
# Mean eastward wind tendency due to parametrisations
# Mean northward wind tendency due to parametrisations

# surface pressure 
# ??

# date: 20200201 to 20200212
# time: 3hourly
# domain: 11.5N, 55.5W, 15N, 60W]
# levels: ?? to 137




#!/bin/ksh

#SBATCH --job-name=mars_eureca
#SBATCH --workdir=/scratch/rd/paaa/mars/bin/

date=20200118
time=0
stream=lwda
for exp in hnl1;do
  if [ $exp == hnl1 ]; then
    stream=oper
  fi
  while [ $date -le 20200215 ]; do
cat<<EOF>mm
RETRIEVE,
    CLASS      = od,
    CLASS      = rd,
    TYPE       = fc,
    STREAM     = $stream,
    anoffset=9,
    EXPVER     = $exp,
    REPRES     = GG,
    LEVTYPE    = ml,
    levelist=80/to/137,
    param=T/q/u/v,
    DATE       = $date,
  # DATE       = 20200118/to/20200215,
    time=$time,
    step=24/to/48/by/3,
    grid=av,
    grid=0.1/0.1,
    area=15/300/11/305,
    DOMAIN     = G,
    TARGET     = "/scratch/rd/paaa/mars/bin/eureca_${exp}_$date.grb"
EOF
mars mm
   grib_to_netcdf -R   $date eureca_${exp}_$date.grb -o eureca_${exp}_$date.nc
   date=$( newdate -D $date +1)
   done # loop dates
done # loop experiments
