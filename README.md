# BlinkViz

BlinkViz is an approximate visualization approach by leveraging Mixed Sum-Product Networks to learn data distribution and corporating Neural Networks to enhance the performance. 

This is the implementation described in (pdf [link](https://doi.org/10.1145/3543507.3583411)).

## Setup

The packed environment could be reproduced by running:

````
conda env create -f environment.yml
````

## Training

The two modules are trained seperately. 

1. To train several SPNEnsembles according to your datasets. Refer to the wiki of DeepDB [link](https://github.com/DataManagementLab/deepdb-public). You could choose to execute the commands of DeepDB, or directly run 

    ````
    bash run_train_spn.sh
    ````


2. To train the neural network by running: 

    ````
    cd res_nn
    bash run_train.sh [job name]
    ````
    


## Evaluation

We achieved an end-to-end approximate query processing. You can choose to evaluate single query and get the answer immediately or to evaluate a batch of queries stored in a file and output the answers to a file.

### Single Query

You could evalute single query by

````
bash run_single_query.sh
````

The interface will prompt you to enter a query and return approximate query results.

### A Batch of Queries

You could evalute a batch of queries by

````
bash run_batch_queries.sh
````

The results will be stored in a CSV file. 

## Visualization

We also provide simple visualization options. You can choose to visualize the approximate answers in bar chart, line chart or pie chart by set options 'visualize=True' and 'visType=\[bar|line|pie\]' in run_single_query.sh.