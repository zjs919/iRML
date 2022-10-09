# iRML-Reasoning
The implementation of iRML reasoning is based on the code released by https://github.com/xwhan/DeepPath.

## About the Dataset
We use the [FB15k-237](https://drive.google.com/file/d/1klWL11nW3ZS6b2MtLW0MHnXu-XlJqDyA/view?usp=sharing) released by the original [DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning](https://arxiv.org/abs/1707.06690) paper. `Semantic_Layering.ipynb` is used for grouping the entities of the dataset into different semantic layers. 

## How to run the code 
1. unzip the data, put the data folder in the `iRML_Reasoning` directory
2. run the following scripts under `scripts/`
    *   `./pathfinder.sh ${relation_name}`  # find the reasoning paths
    *   `./iRML.sh ${relation_name}` # see the numerical results
3. Put `Semantic_Layering.ipynb` into the corresponding task folder and run. Replace the `train_pos` file with the output file, and run `./iRML.sh ${relation_name}` again to see the numerical results considering semantic hierarchy. 
