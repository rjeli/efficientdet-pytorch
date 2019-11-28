#!/bin/bash
set -euo pipefail

gcloud beta compute instances create gpu-trainer \
	--zone=us-west1-a \
	--machine-type=n1-standard-16 \
	--preemptible \
	--accelerator=type=nvidia-tesla-v100,count=4 \
	--image=c2-deeplearning-pytorch-1-3-cu100-20191112
	--image-project=ml-images \
	--boot-disk-size=100GB --boot-disk-type=pd-ssd --boot-disk-device-name=gpu-trainer
