# Noisy-Labels-EPD
Proof-of-concept code for EPD (https://arxiv.org/abs/2009.11128) on MNIST:

We use the algorithm from https://arxiv.org/abs/2009.11128 to learn and perform classification under uniform label noise (symmetry 0.5)

# EPD Parameters

An important question regarding the proposed method has to do with parameter tuning. We identify 3 as parameters as particularly important. To follow the discussion please read the preprint on the link above or our IEEE Access paper:

## How many base models will we use on the deep ensemble?

This is a simple parameter to tune. Intuitively the more the better. However as expected, including more models after a certain amount will not provide  substantial performance imporvements. This amount generally is dependent of the next parameter.

## How much  of the train data will we hold-out during training of the ensemble?

To form the EPs we need to hold-out part of the training data in order to generalize on it with the current base classifier. Ideally we would hold out only one datapoint. However due tocomputational and time constraints this is not feasible. As such a compromise should be struck between execution time and amount of training data to be used on each iteration.

Based on this discussion, if we had "infinite" time or computational resources we would use a huge number of base models trained on all data with the exception of a single randomly chosen datapoint.  

## How many epochs should we train the base classifiers for?

Empirically the amount of epochs to train the base models is one of the most important parameters. Given the noisy nature of the labels, if we do not under-train, the model will focus on erroneous details corresponding to wrong labels and generalization performance will drop.

But how can we identify the appropriate number of epochs to train?

To achieve this we explore 2 different metrics: 
(1)  cross-entropy loss on a held-out validation dataset (with noisy labels)(i.e., choose the amount of epochs that minimizes this noisy val loss)
(2)  the entropy of the EPs after training, assuming different predefined amounts of epochs.(e.g., choose to train the base classifiers for 3,5,8 and 10 epochs. Crerate the respective EPs. Given that all other params are identical choose thenumber of epochs that yield the max EP entropy value)

The intuition for (1) is quite simple: By learning a correct pattern a model can perform correct prediction on the correctly labelled data of the noisy val set. This would transfer when performing generalization for the EPs and as such would hopefully result  into a more correct labelling schema.

For (2) the intuition is two-fold: If we train our base models for too few epochs, the will not learn the correct classification paterns, and as such would disagree during the formation of the EPs, resulting in low EP entropy. On the other hand, if we overtrain, the models will learn more complex boundaries due to the label noise. This would presumably result in low entropy for multiple regions of the input space for most models that should normally b of high entropy. This uncertainty would again be mapped onto the EPs due to the generalization-based proccess followed for their formation. 

For symmetrical noise 0.5 on MNIST both approaches identify similar epochs as good eppochs for training (see .ods file). However more work is needed to properly identify whether these metrics are reliable.

