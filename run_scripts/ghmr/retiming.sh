cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

# JAMES_RESULT=/home/groups/syyeung/jmhb/bio-pose/mdm/save/
# pretrained_model=$JAMES_RESULT/20230304_trainemp_exp19-0-amasshml_augment_FcShapeAxyzAvel-amasshml_augment_FcShapeAxyzAvel/000003/model000410000.pt


pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000059/model000585000.pt


for data in amasshml_FcShapeAxyzAvel; do 

expname=${cur_fname}-${db}-${data}

seed=1
num_samples=100
savedir="test4/0_100_1/"
rollout=4

indir="diffuseIK_itr0.npy"

printf "seed: %s\n" "${seed}"
printf "savedir: %s\n" "${savedir}"


cmd4="python -m generative_infill.retiming \
		--data_config_path gthmr/emp_train/config/data/${data}.yml \
                --save_dir gthmr/results/${expname} \
                --model_path $pretrained_model 
"

echo $cmd4
eval $cmd4
echo "Elapsed Time interpolate: $SECONDS"

done
