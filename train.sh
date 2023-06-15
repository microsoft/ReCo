export TORCH_HOME="~/.torch"

### COCO
python main.py --base configs/reco/v1-finetune_cocogit.yaml -t --nodes 4 --actual_resume logs/sd-v1-4-full-ema.ckpt -n coconoreg_maxboxGIT_1e4n4b8c8_aug_L616 --gpus 0,1,2,3,4,5,6,7 --data_root dataset/coco/coco_image_gen_origin_id/coco_vqgan_train_new.tsv --val_data_root dataset/coco/coco_image_gen_origin_id/coco_vqgan_full_test_new.tsv --no-test true