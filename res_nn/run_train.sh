name=$1 # job name
python -u train.py --name $name --device "cuda:0"
