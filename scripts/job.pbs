#!/bin/bash
#PBS -l nodes=8:ppn=20
#PBS -l walltime=12:00:00
#PBS -l mem=350GB
#PBS -N turnstile-{idhash}
#PBS -m ae
#PBS -M danfm@nyu.edu
#PBS -j oe
 
module purge
export PATH="$HOME/anaconda/bin:$PATH"

# Needed environment variables.
export KPLR_DATA_DIR=$SCRATCH
export TURNSTILE_PATH=$WORK/turnstile/cache

# Locations.
export SRCDIR=$HOME/projects/turnstile
export RUNDIR=$WORK/turnstile/bright_g_stars
export PROFILEDIR=$RUNDIR/profile-{idhash}
mkdir -p $RUNDIR
cd $RUNDIR

# Set up and start the IPython cluster.
cp -r $HOME/.ipython/profile_mpi $PROFILEDIR
ipcluster start -n $PBS_NP --profile-dir=$PROFILEDIR &> ipcluster-{idhash}.log &

sleep 5
for (( try=0; try < 100; ++try )); do
    if cat ipcluster-{idhash}.log | grep -q "Engines appear to have started successfully"; then
        success=1
        break
    fi
    sleep 5
done

if (( success )); then
    # Run the analysis.
    python $SRCDIR/scripts/search.py $RUNDIR 2000 $SCRATCH/data/lightcurves --ninj 5 --profile-dir $PROFILEDIR &> output-{idhash}.log
else
    echo "Server never started" &> output-{idhash}.log
fi

# Shut the cluster down.
ipcluster stop --profile-dir=$PROFILEDIR

exit $(( 1-success ));
