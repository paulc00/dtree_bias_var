# dtree_bias_var
Plot bias, variance and overall accuracy for a boosted ID3 decision tree on the SPECT Heart dataset.

This implements a multi-class [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) decision tree using Information Gain 
as the fit function and a Chi-square test and/or max tree depth to determine when to prune. 
See [Induction of Decision Trees, Ross Quinlan, 1986](http://hunch.net/~coms-4771/quinlan.pdf) for details.

dtree_test.py initially trains the full training set then runs a test and prints the accuracy.
It then goes on to train using 25 bootstrap samples (bagging), repeatedly for tree depths from 1 to 10.

The bias, variance and overall accuracy is then plotted against tree depth.

Accuracy on the small dataset provided is about 71% which seems fairly reasonable. The accuracy had been at a little over 60% before
@surajrautela pointed out an error in the Information Gain calculation.
