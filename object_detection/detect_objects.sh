export OD_DIR='/data/local/vw120/aokvqa_base/od'
od_split=val
split_dir="${OD_DIR}/${od_split}2017"

echo "Creating split_dir: ${split_dir}"
mkdir -p ${split_dir}

python object_detection/detect_objects.py \
    --aokvqa-dir ${AOKVQA_DIR} \
    --coco-dir ${COCO_DIR} \
    --split ${od_split} \
    --model faster_rcnn \
    --batch-size 1 \
    --out-dir ${split_dir}