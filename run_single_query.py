import pickle
from Ensemble import Ensemble
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spns', default=['/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles_01/ensemble_single_flights500M_10000000.pkl','/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles_02/ensemble_single_flights500M_10000000.pkl','/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles_03/ensemble_single_flights500M_10000000.pkl'])
    parser.add_argument('--nn', default="/home/qym/spn_ensemble/logs/resd6h512i50w_3model/base/hidden_dim[512]spn_input_dims[[3, 3, 3]]useTree[False]depth[6]use_norm[False](08_17_07_10)/final.pth")
    parser.add_argument('--dataset', default="flights500M")
    parser.add_argument('--query', default="SELECT year_date, COUNT(*) FROM flights GROUP BY year_date;")
    #parser.add_argument('--dataset_batch', default="/home/qym/datasets/single_queries/")
    args = parser.parse_args()

    spns = args.spns
    nn = args.nn
    dataset = args.dataset
    query = args.query

    ensemble = Ensemble(spns, nn, dataset)
    result = ensemble.evaluate_together_single_query(query)
