# BlinkViz

## Training

The two modules are trained seperately. 

1. To train several SPNEnsembles according to your datasets. Refer to the wiki of DeepDB(link).

2. To train the neural network by running: cd res_nn; bash run_train.sh [job name]. Don't forget to check the configs.

## Evaluating

We achieved an end-to-end approximate query processing. You can choose to evaluate single query and get the answer immediately or to evaluate a batch of queries stored in a file and output the answers to a file.

### Single Query

### A Batch of Queries

## Visualization

We also provide simple visualization options. You can choose to visualize the approximate answers in bar chart, line chart or pie chart.