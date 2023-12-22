# BlinkViz

BlinkViz is an approximate visualization approach by leveraging Mixed Sum-Product Networks to learn data distribution and corporating Neural Networks to enhance the performance. 

This is the implementation described in (pdf [link](https://doi.org/10.1145/3543507.3583411)).

## Setup

The packed environment could be reproduced by running:

````
conda env create -f environment.yml
````

## Training

To train several SPNEnsembles according to your datasets. Refer to the wiki of DeepDB [link](https://github.com/DataManagementLab/deepdb-public). You could choose to execute the commands of DeepDB, or directly run 

````
bash run_train_spn.sh
````

To train the neural network by running: 

````
cd res_nn
bash run_train.sh [job name]
````

## Evaluation

To evaluate the SPNEnsembles of Flights dataset by running the AQP evaluation commands of DeepDB

````
python3 maqp.py --evaluate_aqp_queries
    --dataset flights500M
    --target_path ./baselines/aqp/results/deepDB/flights500M_model_based.csv
    --ensemble_location ./flights-benchmark/spn_ensembles/ensemble_single_flights500M_10000000.pkl
    --query_file_location ./benchmarks/flights/sql/aqp_queries.sql
    --ground_truth_file_location ./benchmarks/flights/ground_truth_500M.pkl  
    --target_ns_path ../flights-benchmark/spn_ensembles/tmp_ns_file
````

To evaluate the residual network, original SQL with GROUP-BY clause should be converted into many SQL statements without GROUP-BY clause and get their corresponding SPN predictions as inputs. For example, if there is a query like

````
SELECT DEST, COUNT(*) FROM flights WHERE YEAR_DATE=2000 GROUP BY DEST;
````

it should be first decomposed into

````
SELECT DEST, COUNT(*) FROM flights WHERE YEAR_DATE=2000 AND DEST='JFK';
SELECT DEST, COUNT(*) FROM flights WHERE YEAR_DATE=2000 AND DEST='OW';
````

which exist as two training data records of the neural networks.

The format of training data of residual network module is

````
[SPN1 status vector] [SPN2 status vector] [SPN3 status vector] [SPN1 prediction] [SPN2 prediction] [SPN3 prediction] [ground truth]
````


And then run the evaluation by

````
cd res_nn
bash run_evaluate.sh
````

## Citation

Yimeng Qiao, Yinan Jing, Hanbing Zhang, Zhenying He, Kai Zhang, and X. Sean Wang. 2023. BlinkViz: Fast and Scalable Approximate Visualization on Very Large Datasets using Neural-Enhanced Mixed Sum-Product Networks. In Proceedings of the ACM Web Conference 2023 (WWW '23). Association for Computing Machinery, New York, NY, USA, 1734–1742. https://doi.org/10.1145/3543507.3583411
