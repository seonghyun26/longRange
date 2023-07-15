
for dataset in "peptides-func" "peptides-struct"
do
  for model in "GCNII" "GINE"
  do
    python main.py --cfg configs/LGNN/$dataset-LGNN-$model.yaml  wandb.use True  wandb.project lrgb  gnn.lgvariant 6 &
    sleep 10
  done
done