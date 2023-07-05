DATASET="peptides-func"

for i in 3 4 5 6 7
do
  # echo $i
  python main.py --cfg configs/GCNII/peptides-func-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  python main.py --cfg configs/LGNN/peptides-func-LGNN-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
done
