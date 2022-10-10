# Collaborative Reasoning

The implementation of collaborative reasoning is based on the code released by https://github.com/yh-yao/FedGCN. 

## Data

2 datasets were used in the paper:

- Cora
- Citeseer

The Pubmed dataset is also included, but the results are not presented in the paper due to the page limit. 

## Requirements
  * Python 3
  * PyTorch
  * networkx
  * numpy

## How to Use
* Run the `Collaborative_Reasoning_p.ipynb` file to see the numerical results for different vaules of $p$. 
* Run the `Collaborative_Reasoning_NumDevice.ipynb` file to see the numerical results for different numbers of devices. 
