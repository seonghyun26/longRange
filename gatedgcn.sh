for dataset in "peptides-func"
# "peptides-struct"
do
  for i in 5 6 7 8 9
  do
    python main.py --cfg configs/LGNN/$dataset-LGNN-GatedGCN.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  done
done
