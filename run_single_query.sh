spns=('/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles/ensemble_single_flights500M_01.pkl','/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles/ensemble_single_flights500M_02.pkl','/home/qym/github_keeping/BlinkViz/flights-benchmark/spn_ensembles/ensemble_single_flights500M_03.pkl')
nn="/home/qym/spn_ensemble/logs/resd6h512i50w_3model/base/hidden_dim[512]spn_input_dims[[3, 3, 3]]useTree[False]depth[6]use_norm[False](08_17_07_10)/final.pth"
dataset='flights500M'
query='SELECT year_date, COUNT(*) FROM flights GROUP BY year_date;'

python3 run_single_query.py --spns $spns --nn $nn --dataset $dataset --query $query