# BlinkViz

BlinkViz is an approximate visualization approach by leveraging Mixed Sum-Product Networks to learn data distribution and corporating Neural Networks to enhance the performance. 

This is the implementation described in (pdf link).

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
    
    Don't forget to check the configs.

3. To combine the two model together by

    ````
    bash run_generate_ensemble.sh
    ````

## Evaluation

We achieved an end-to-end approximate query processing. You can choose to evaluate single query and get the answer immediately or to evaluate a batch of queries stored in a file and output the answers to a file.

### Single Query

You could run the single query script and enter the interactive interface

````
bash run_single_query.sh
````

The interface will prompt you to enter a query and return approximate query results.

### A Batch of Queries

You could run the batch query script and put your queries in the specified location. Run

````
bash run_batch_queries.sh
````

The query results will exist in a CSV file.

## Visualization

We also provide simple visualization options. You can choose to visualize the approximate answers in bar chart, line chart or pie chart.

Please refer to our testbed on the Github (coming soon).