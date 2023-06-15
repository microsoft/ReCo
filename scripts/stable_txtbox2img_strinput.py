import argparse, os, sys, glob, re
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPTokenizer, CLIPTextModel

def pre_caption(caption, max_words):
    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference-box.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")

    if opt.config != 'configs/stable-diffusion/v1-inference.yaml':
        max_src_length = 1232 if '1232' in opt.ckpt else 616
        config.model.params.cond_stage_config.params['extend_outputlen']=max_src_length
        config.model.params.cond_stage_config.params['max_length']=max_src_length

    model = load_model_from_config(config, f"{opt.ckpt}")
    #model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))


    ## box processor prepare
    cliptext_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    box_offset = len(cliptext_tokenizer)
    coco_hw = json.load(open('dataset/coco_wh.json', 'r'))
    coco_od = json.load(open('dataset/coco_allbox_gitcaption.json', 'r')) ## 'tag' 'pad_caption' 'crop_caption'
    num_bins, centercrop = 1000, True
    num_withbox = 0
    for key in coco_od:
        if 'box' in coco_od[key]:
            if len(coco_od[key]['box'])!=0: num_withbox+=1
    ## end of box processor prepare

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    seed_everything(opt.seed+n)
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        ##
                        token_prompt = []
                        prompt_seq = []
                        for x in prompts[0].split(' <'):
                            prompt_seq += [y for y in x.split('>') if y!='' and y!=' ']

                        tokenized_text = []
                        str_caption = prompt_seq[0]
                        text = [pre_caption(prompt_seq[0].lower(), max_src_length)]
                        text_enc = cliptext_tokenizer(text, truncation=True, max_length=max_src_length, return_length=True,
                                                    return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                        tokenized_text.append(text_enc[0,:])
                        # tokenized_text = []
                        # str_caption = ''

                        for ii in range(1,len(prompt_seq),5):
                            box, caption = [float(x) for x in prompt_seq[ii:ii+4]], prompt_seq[ii+4][1:]

                            box = [float(x)/num_bins for x in box]
                            quant_x0 = int(round((box[0] * (num_bins - 1)))) + box_offset
                            quant_y0 = int(round((box[1] * (num_bins - 1)))) + box_offset
                            quant_x1 = int(round((box[2] * (num_bins - 1)))) + box_offset
                            quant_y1 = int(round((box[3] * (num_bins - 1)))) + box_offset
                            region_coord = torch.tensor([quant_x0,quant_y0,quant_x1,quant_y1]).to(text_enc.device)
                            caption = pre_caption(caption.lower(), max_src_length)
                            region_text = cliptext_tokenizer(caption, truncation=True, max_length=max_src_length, return_length=True,
                                                        return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                            tokenized_text.append(region_coord)
                            tokenized_text.append(region_text[0,:])
                            str_caption += ' <%d> <%d> <%d> <%d> '%(quant_x0-box_offset,quant_y0-box_offset,quant_x1-box_offset,quant_y1-box_offset) + caption
                        tokenized_text = torch.cat(tokenized_text, dim=0)[:max_src_length]
                        pad_tokenized_text = torch.tensor([box_offset-1]*max_src_length).to(text_enc.device)
                        pad_tokenized_text[:len(tokenized_text)] = tokenized_text
                        prompts = pad_tokenized_text.unsqueeze(0)
                        opt.prompt, prompt = str_caption[:200], str_caption[:200]

                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # Image.fromarray(x_sample.astype(np.uint8)).save(
                                #     os.path.join(sample_path, f"{base_count:05}.jpg"))
                                repeat=0
                                while os.path.isfile(os.path.join(outpath, "%s_%d.png"%(prompt,repeat))):
                                    repeat+=1
                                pred_image = Image.fromarray(x_sample.astype(np.uint8))
                                pred_image.save(os.path.join(outpath, "%s_%d.png"%(prompt,repeat)))
                                base_count += 1


                        draw = ImageDraw.Draw(pred_image)
                        for ii in range(1,len(prompt_seq),5):
                            box, caption = [float(x) for x in prompt_seq[ii:ii+4]], prompt_seq[ii+4][1:]
                            box = [float(x)/num_bins for x in box]
                            pred_dim = pred_image.size[0]
                            left, top, right, bottom = int(box[0]*pred_dim), int(box[1]*pred_dim), int(box[2]*pred_dim), int(box[3]*pred_dim)
                            left, top, right, bottom = min(max(0,left),pred_dim), min(max(0,top),pred_dim), min(max(0,right),pred_dim), min(max(0,bottom),pred_dim)
                            if left>=right or top>=bottom:
                                continue
                            draw.rectangle(((left, top), (right, bottom)), fill=None, outline=(184,134,11), width=int(8*512/500))
                        pred_image.save(os.path.join(outpath, "bbox/%s_%d.png"%(prompt,repeat)))

                    

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    print(pad_tokenized_text)


if __name__ == "__main__":
    main()
