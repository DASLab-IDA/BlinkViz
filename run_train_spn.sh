dataset='flights500M'
csv_seperator=','
csv_path='../flights-benchmark'
hdf_path='../flights-benchmark/gen_hdf_01'
ensemble_strategy='single'
ensemble_path='./flights-benchmark/spn_ensembles_01'
rdc_threshold=0.3

python3 -Xfaulthandler maqp.py --generate_hdf --dataset $dataset --csv_seperator $csv_seperator --csv_path $csv_path --hdf_path $hdf_path
python3 -Xfaulthandler maqp.py --generate_ensemble --dataset $dataset --samples_per_spn 10000000 --ensemble_strategy $ensemble_strategy --hdf_path $hdf_path --ensemble_path $ensemble_path --rdc_threshold $rdc_threshold --post_sampling_factor 10