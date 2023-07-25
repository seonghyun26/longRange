
# config1="peptides-func-LGNN-GCN+LapPE.yaml"
# python main.py --cfg configs/LGNN/$config1  wandb.use True  wandb.project lrgb  &
# sleep 10

# config2="peptides-func-LGNN-GINE+LapPE.yaml"
# python main.py --cfg configs/LGNN/$config2  wandb.use True  wandb.project lrgb  &
# sleep 10

# config3="peptides-func-LGNN-GatedGCN+LapPE.yaml"
# python main.py --cfg configs/LGNN/$config3  wandb.use True  wandb.project lrgb  &
# sleep 10

# GCN GINE GatedGCN

config="peptides-func-LGNN-"$1"+LapPE.yaml"
echo $config
for layer in 7 9 11 13 15 17 19 21 23
do
  python main.py --cfg configs/LGNN/$config  wandb.use True  wandb.project lrgb  gnn.layers_mp $layer
done