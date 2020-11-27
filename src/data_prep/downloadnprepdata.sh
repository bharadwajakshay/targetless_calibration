#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

array=( 0001 0002 0003)

task ()
{
    wget -k -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_$1/2011_09_26_drive_$1_sync.zip
    unzip 2011_09_26_drive_$1_sync.zip
    echo "Extracting $1"

    python -B $SCRIPTPATH/decalibration.py $PWD/2011_09_26/2011_09_26_drive_$1_sync/
}

# Download the calibration file 

for id in {0..4}
do
   ((i=i%N)); ((i++==0)) && wait
   task ${array[$id]}
done

wait
 echo "processing done"