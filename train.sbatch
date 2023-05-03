#!/bin/bash
#
#SBATCH --job-name=4fcv_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=48GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc8895@nyu.edu
#SBATCH --output=cvtrain4_fscratch.out

module purge

singularity exec --nv\
	    --overlay /scratch/mc8895/overlay-25GB-500K-1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate pt2; \
        python train.py \
            --x=4 \
            --save_path=decoder_scratch \
            --epoch=10"