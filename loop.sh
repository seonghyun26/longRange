
DATASET="peptides-func"


for model in "GCNII" "GINE" "GatedGCN" 
do
  python main.py --cfg configs/LGNN/$DATASET-LGNN-$model.yaml  wandb.use True  wandb.project lrgb  gnn.lgvariant 3
done

# sleep 10
# python main.py --cfg configs/GINE/$DATASET-GINE.yaml  wandb.use True  wandb.project lrgb &


# python main.py --cfg configs/LGNN/peptides-func-LGNN.yaml  wandb.use True  wandb.project lrgb 
# python main.py --cfg configs/GINE/peptides-func-GINE.yaml  wandb.use True  wandb.project lrgb 

# python main.py --cfg configs/LGNN/peptides-func-LGNNGatedGCN.yaml  wandb.use True  wandb.project lrgb 
# python main.py --cfg configs/LGNN/peptides-func-LGNN-GatedGCN+RWSE.yaml  wandb.use True  wandb.project lrgb 