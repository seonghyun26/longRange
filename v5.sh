# DATASET="peptides-struct"
DATASET="peptides-func"

for model in "GCNII" "GINE" "GatedGCN"
do
  python main.py --cfg configs/LGNN/$DATASET-LGNN-$model.yaml  wandb.use True  wandb.project lrgb  gnn.lgvariant 5 &
  sleep 10
done