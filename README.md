# Thesis Appendix
This github repository contains the appendix of my masters' thesis.
The appendix covers the python code for TensorForce and Stable Baselines implementations, in addition to data behind included graphs and the applied motion data after preprocessing.

## Code
The code in this appendix is collected in the `Code` sub-directory. This directory contains two sub-directory, `Baselines` and `TensorForce`, which contain respectively the Stable Baselines and TensorForce implementations. The `Code` directory also contains two docker files for building a docker environment to run the code with. 
`Dockerfile.baseline` should be build with the tag `remimic:baselines` and 
`Dockerfile.TensorForce` with the tag `remimic:mpi-cpu-prebuildbullet`.
To run a training session, run the command: 

	docker run -v DIRPATH:/usr/src/app TAG

Here `DIRPATH` should be the path to the `Baselines` or `TensorForce` sub-directory, and `TAG` the appropriate docker image tag. 

The training configurations are edited through the `baselines_main.py` and `main.py` files in the `Baselines` and `TensorForce` directories respectively. Threshold limiting needs to be manually adjusted in the `Baselines/Environment/baselines_pbenv.py` file line 147, or `TensorForce/Environment/tforce_env_full_state.py` line 143.
As this is not a computer science thesis, the code has not been cleaned up, and is provided only for the sake of completion.

## Training Data
Collected data for the graphs in the thesis is included in the subdirectory `Training Data`. Data is in the `csv` file format, labeled with the figure numbers from the thesis, followed by line label, eg. `2a Threshold .4.csv`. Files contain two columns, the first containing average performance, and the second the episode number. 

## Motion Data
The motion data used in this thesis can be found in both the Stable Baselines and TenorForce code directories under the names `Trial01.npy` and `Trial09.npy`, containing the static and walking motions respectively. Note that only these two preprocessed data files are included, as the unprocessed data is about 1GB in size, and contains sensitive personal data of the human actors.