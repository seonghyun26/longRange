for dataset in "peptides-struct"
# "peptides-struct"
do
  for model in "GCNII" "GINE" "GatedGCN"
  do
    python main.py --cfg configs/LGNN/$dataset-LGNN-$model.yaml  wandb.use True  wandb.project lrgb &
    sleep 10
  done
done

for model in "GCNII" "GINE" "GatedGCN"
do
  python main.py --cfg configs/$model/peptides-struct-$model.yaml  wandb.use True  wandb.project lrgb 
done