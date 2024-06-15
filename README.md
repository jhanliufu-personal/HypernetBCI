# HypernetBCI
Quick calibration of DL-based BCI models enabled by hypernetworks

## Training scripts
* {MI, SS}_baseline_1(_torch): baseline 1 experiments are the ones that train a model FROM SCRATCH for each person using different amount of training data. 

* {MI, SS}_baseline_2(_torch): baseline 2 experiments are the ones that take each person as the hold out / new person and pretrain
the BCI model on everyone else together as a big pretrain pool. No distinction between people in the pretrain model.
