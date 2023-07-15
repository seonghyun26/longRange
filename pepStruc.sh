for dataset in "peptides-struct"
# "peptides-struct"
do
  for model in "GCNII" "GINE" "GCN"
  do
    for version in 3 4 5 6 
    do
      python main.py --cfg configs/LGNN/$dataset-LGNN-$model.yaml  wandb.use True  wandb.project lrgb  gnn.lgvariant $version &
      sleep 10
    done
    python main.py --cfg configs/LGNN/$dataset-LGNN-GatedGCN.yaml  wandb.use True  wandb.project lrgb  gnn.lgvariant $version
  done
done

# for model in "GCNII" "GINE" "GatedGCN"
# do
#   python main.py --cfg configs/$model/peptides-struct-$model.yaml  wandb.use True  wandb.project lrgb 
done