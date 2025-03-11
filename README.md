# tgfrog
Code related to the implementation of TG-FROG in Colbert

The code can be installed as a python library:

    pip install -e .

The main entry point to get a FROG spectrogram is in 
> src/tgfrog/signal_processing/frog_simulator.py

An inverse solver (find the pulse parameters given a target spectrogram) is in
> src/tgfrog/solver/scipy_optimize.py

This is a solver based on scipy optimize.