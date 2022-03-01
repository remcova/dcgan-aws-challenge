## My entry for the AWS DL1 Challenge by Devpost
Author: Remco van Akker
<br />

# Project name
## Using a Deep Convolutional GAN Network to Generate Synthetic Data for Medical Purposes based on Real-Life Data
<br />

## Usage instructions
### Train with HPU
If you want to make use of HPU, run the main script with the following command: \
python3 dcgan.py --use_hpu

### Distributed Training with Horovod
If you want to make use of Distributed Training via Horovod (up to 8 Gaudi cores), run the main script with the following command: \
mpirun -np 8 python3 dcgan.py --use_hpu

### Restore Checkpoint
You can resume from the latest checkpoint by running the script with the following command:
python3 dcgan.py --use_hpu --restore

### Download Dataset
You don't have to download anything manual, just replace the kaggle.json file with your own and the application will use your Kaggle API Key to download the dataset.

Dataset source: https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k
