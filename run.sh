DATASET="peptides-func"
# DATASET="vocsuperpixels"


# python main.py --cfg configs/GatedGCN/vocsuperpixels-GatedGCN.yaml  wandb.use True  wandb.project lrgb  device 2

# python main.py --cfg configs/GatedGCN/peptides-func-GatedGCN.yaml  wandb.use True  wandb.project lrgb
# python main.py --cfg configs/GCN/peptides-func-GCN.yaml  wandb.use True  wandb.project lrgb &
# python main.py --cfg configs/GCNII/peptides-func-GCNII.yaml  wandb.use True  wandb.project lrgb &
# python main.py --cfg configs/GINE/peptides-func-GINE.yaml  wandb.use True  wandb.project lrgb &
python main.py --cfg configs/LGNN/peptides-func-LGNN.yaml  wandb.use True  wandb.project lrgb &
# python main.py --cfg configs/GT/peptides-func-Transformer+LapPE.yaml  wandb.use True  wandb.project lrgb &
# python main.py --cfg configs/SAN/peptides-func-SAN.yaml  wandb.use True  wandb.project lrgb &


# python main.py --cfg configs/GCN/$DATASET-GCN.yaml  wandb.use True  wandb.project lrgb   device 0 &
# python main.py --cfg configs/GCNII/$DATASET-GCNII.yaml  wandb.use True  wandb.project lrgb   device 1 &
# python main.py --cfg configs/GINE/$DATASET-GINE.yaml  wandb.use True  wandb.project lrgb   device 2 &
# python main.py --cfg configs/GatedGCN/$DATASET-GatedGCN.yaml  wandb.use True  wandb.project lrgb   device 3 &

# python main.py --cfg configs/GT/$DATASET-Transformer+LapPE.yaml  wandb.use True  wandb.project lrgb   device 4 &
# python main.py --cfg configs/SAN/$DATASET-SAN.yaml  wandb.use True  wandb.project lrgb  &
