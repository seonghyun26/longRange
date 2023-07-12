for dataset in "peptides-func"
# "peptides-struct"
do
  for i in 4 5 6 7
  do
    python main.py --cfg configs/GCNII/$dataset-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
    python main.py --cfg configs/LGNN/$dataset-LGNN-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  done
done

# dataset="peptides-func"
# python main.py --cfg configs/LGNN/$dataset-LGNN-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp 6
# python main.py --cfg configs/LGNN/$dataset-LGNN-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp 7
