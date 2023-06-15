export TORCH_HOME="~/.torch"

ckpt=logs/reco_laion_1232.ckpt
# ckpt=logs/reco_coco_616.ckpt

GEN_IMAGE_PATH=outputs/reco_laion
# GEN_IMAGE_PATH=outputs/reco_coco

mkdir -p outputs
mkdir -p $GEN_IMAGE_PATH
mkdir -p ${GEN_IMAGE_PATH}/top1
mkdir -p ${GEN_IMAGE_PATH}/coco_crop
mkdir -p ${GEN_IMAGE_PATH}/top1_crop
mkdir -p ${GEN_IMAGE_PATH}/top1_crop_foldered

python scripts/scenefid_txtbox2img_dataset_inference.py --plms --n_samples 1 --n_iter 1 --scale 4.0 --ddim_steps 50  --ckpt $ckpt --from-file dataset/coco/coco_image_gen_origin_id/coco_vqgan_full_test_new.tsv --skip_grid --outdir ${GEN_IMAGE_PATH}/top1 --centercrop --box_descp caption
python scripts/draw_box.py --path ${GEN_IMAGE_PATH}/top1

python scripts/cleanfid_call.py --transform center --gpu 0 --path1 ${GEN_IMAGE_PATH}/top1_crop --path2 ${GEN_IMAGE_PATH}/coco_crop
python scripts/cleanfid_call.py --transform center --gpu 0 --path1 ${GEN_IMAGE_PATH}/top1 --path2 dataset/coco/coco_test30k
