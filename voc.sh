dataset="vocsuperpixels"

for model in "GCNII" "GINE" "GatedGCN" 
do
  python main.py --cfg configs/$model/$dataset-$model.yaml  wandb.use True  wandb.project lrgb &
  sleep 10
done

# for model in "GCNII" "GINE" "GatedGCN" 
# do
#   python main.py --cfg configs/LGNN/$dataset-LGNN-$model.yaml  wandb.use True  wandb.project lrgb &
#   sleep 10
# done