# Noisy-Labels-EPD
Proof-of-concept code for EPD on MNIST:

We use the algorithm from https://arxiv.org/abs/2009.11128 to learn and perform classification under uniform label noise (symmetry 0.5)

# EPD Parameters

An important question regarding the proposed method has to do with parameter tuning. We identify 3 as parameters as particularly important:

## How many base models will we use on the deep ensemble?

This is a simple parameter to tune. Intuitively the more the better. However as expected, including more models after a certain amount will not provide  substantial performance imporvements. This amount generally is dependent of the next parameter.

## How much  of the train data will we hold-out during training of the ensemble?

To form the EPs we need to hold-out part of the training data in order to generalize on it with the current base classifier. Ideally we would hold out only one datapoint. However due tocomputational and time constraints this is not feasible. As such a compromise should be struck between execution time and amount of training data to be used on each iteration

## How many epochs should we train the base classifiers for?
