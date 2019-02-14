# Invariant and  Equivariant Graph Networks
A TensorFlow implementation of The ICLR 2019 paper "Invariant and  Equivariant Graph Networks" by Haggai Maron, Heli Ben-Hamu, Nadav Shamir and Yaron Lipman
https://openreview.net/forum?id=Syx72jC9tm
## Abstract
Invariant and equivariant networks have been successfully used for learning images, sets, point clouds, and graphs. A basic challenge in developing such networks is finding the maximal collection of invariant and equivariant \emph{linear} layers. Although this question is answered for the first three examples (for popular transformations, at-least), a full characterization of invariant and equivariant linear layers for graphs is not known.
In this paper we provide a characterization of all permutation invariant and equivariant linear layers for (hyper-)graph data, and show that their dimension, in case of edge-value graph data, is $2$ and $15$, respectively. More generally, for graph data defined on $k$-tuples of nodes, the dimension is the $k$-th and $2k$-th Bell numbers. Orthogonal bases for the layers are computed, including generalization to multi-graph data. The constant number of basis elements and their characteristics allow successfully applying the networks to different size graphs. From the theoretical point of view, our results generalize and unify recent advancement in equivariant deep learning. In particular, we show that our model is capable of approximating any message passing neural network.
Applying these new linear layers in a simple deep neural network framework is shown to achieve comparable results to state-of-the-art and to have better expressivity than previous invariant and equivariant bases.

## Data
Data should be downloaded from: https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?dl=0. 
Run the following commands in order to unzip the data and put its proper path.
```
mkdir data
unzip benchmark_graphs.zip -d data
```

## Code

### Prerequisites

python3

TensorFlow gpu 1.9.0.

Additional modules: numpy, pandas, matplotlib



### Running the tests

The folder main_scripts contains scripts that run different experiments:

1. To run 10-fold cross-validation with our hyper parameters run the main_10fold_experiment.py script. You can choose the datase in 10fold_config.json.
2. To run hyper-parameter search, run the main_parameter_search.py script  with the corresponding config file
3. To run training and evaluation on one of the data sets run the main.py script

example to run 10-fold cross-validation experiment:

```
python3 -m main_10fold_experiment --config=configs/10fold_config.json
```




