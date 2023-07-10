
# for dataset in "peptides-func" "peptides-struct"
for dataset in  "cocosuperpixels" "vocsuperpixels" "pcqm-contact"
do
python er.py --cfg configs/GCNII/$dataset-GCNII.yaml  wandb.use False
# python effecResist.py --cfg configs/LGNN/$dataset-LGNN-GCNII.yaml  wandb.use False
done
