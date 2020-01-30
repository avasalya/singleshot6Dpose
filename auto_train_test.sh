cfg_name=()
cfg_list=()
for file in ./models_cfg/yolo_lite/*; do
    cfg_list=("${cfg_list[@]}" ${file})
    f_name=${file##*/}
    cfg_name=("${cfg_name[@]}" "${f_name%%.*}")
done
# for i in "${cfg_list[@]}"; do echo "$i" ; done

weights_list=()
for file in ./initial_weights/yolo_lite/*; do
    weights_list=("${weights_list[@]}" ${file})
done
# for i in "${weights_list[@]}"; do echo "$i" ; done

objects_list=()
for file in ./objects_cfg/*; do
    objects_list=("${objects_list[@]}" ${file})
done

for ((i=0; i<${#weights_list[@]}; ++i)); do
    for ((j=0; j<${#objects_list[@]}; ++j)); do
        object_name="${objects_list[j]:14:-5}"
        printf "Object %s for %s is in %s\n" "$object_name" "${cfg_name[i]}" "${weights_list[i]}"
        python train.py --datacfg="${objects_list[j]}" --modelcfg="${cfg_list[i]}" --initweightfile="${weights_list[i]}"\
                --distiled=0 --backupdir=./backup/trained/tiny-"$object_name"/"${cfg_name[i]}"/

        python valid.py --datacfg="${objects_list[j]}" --modelcfg="${cfg_list[i]}" --distiled=0 \
                --weightfile=./backup/trained/tiny-"$object_name"/"${cfg_name[i]}"/model.weights
    done
    # python valid.py --datacfg="${objects_list[j]}" --modelcfg=cfg/yolo-pose.cfg \
    #             --weightfile=backup/$object_name/model.weights #gt
done
