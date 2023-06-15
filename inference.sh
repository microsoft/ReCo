export TORCH_HOME="~/.torch"

prompt="Zoomed out view of a man riding a horse through rural country side. <546> <621> <885> <861> brown horse <654> <512> <739> <779> a man in blue shirt"
ckpt=reco_laion_1232.ckpt
# ckpt=reco_coco_616.ckpt
mkdir -p outputs
mkdir -p outputs/bbox
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txtbox2img_strinput.py --plms --n_samples 1 --scale 2.0 --n_iter 4 --ddim_steps 50 --outdir outputs/ --ckpt logs/${ckpt} --prompt "${prompt}"