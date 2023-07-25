for dataset in "peptides-func"
# "peptides-struct"
do
  for i in 7 9 11 13
  do
    python main.py --cfg configs/GatedGCN/$dataset-GatedGCN.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i  &
    sleep 10
    # python main.py --cfg configs/LGNN/$dataset-LGNN-GatedGCN.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  done
done
