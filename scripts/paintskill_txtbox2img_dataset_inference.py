import argparse, os, sys, glob, json, re, random, math
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
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


def process_centercrop(boxes, imagehw, centercrop):
    h, w = float(imagehw[0]), float(imagehw[1])
    if centercrop:
        if w>=h:
            boxes[0] = (boxes[0]+h/2-w/2)/h
            boxes[2] = (boxes[2]+h/2-w/2)/h
            boxes[1], boxes[3] = boxes[1]/h, boxes[3]/h
        else:
            boxes[1] = (boxes[1]+w/2-h/2)/w
            boxes[3] = (boxes[3]+w/2-h/2)/w
            boxes[0], boxes[2] = boxes[0]/w, boxes[2]/w
    else:
        boxes[0], boxes[2] = boxes[0]/w, boxes[2]/w
        boxes[1], boxes[3] = boxes[1]/h, boxes[3]/h
    boxes[0], boxes[1] = max(0., boxes[0]), max(0., boxes[1])
    boxes[2], boxes[3] = max(0., boxes[2]), max(0., boxes[3])
    boxes[0], boxes[1] = min(1., boxes[0]), min(1., boxes[1])
    boxes[2], boxes[3] = min(1., boxes[2]), min(1., boxes[3])
    return boxes

#####################################
## 32*32 / (640*480) = 0.003333333
## 96*96 / (640*480) = 0.03
def process_spatial_word(box, caption, spatial_word_mode='all'):
    if spatial_word_mode == 'tag' or spatial_word_mode == 'caption':
        return '%s '%caption
    box_w, box_h = box[2]-box[0], box[3]-box[1]
    aspect, size = box_w / box_h, box_w*box_h
    box_cx, box_cy = (box[2]+box[0])/2., (box[3]+box[1])/2.
    size_word, aspect_word, location_word, tag_word = '', '', '', ''
    tag_word = caption

    if size<0.003333333: size_word = 'small'
    elif size>0.03: size_word = 'large'
    else: size_word = 'medium'

    if aspect>2.: aspect_word = 'long'
    elif aspect<0.5: aspect_word = 'tall'
    else: aspect_word = 'square'

    if box_cx<(1./3):
        if box_cy<(1./3): location_word = 'top left'
        elif box_cy>(2./3): location_word = 'bottom left'
        else: location_word = 'left'
    elif box_cx>(2./3):
        if box_cy<(1./3): location_word = 'top right'
        elif box_cy>(2./3): location_word = 'bottom right'
        else: location_word = 'right'
    else:
        if box_cy<(1./3): location_word = 'top'
        elif box_cy>(2./3): location_word = 'bottom'
        else: location_word = 'center'
    # prompt = '%s %s %s in the %s.'%(size_word, aspect_word, tag_word, location_word)
    prompt = '%s %s in the %s, %s '%(size_word, aspect_word, location_word, tag_word)
    if spatial_word_mode == 'all':
        return prompt
        
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

    # model.cuda()
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
        "--centercrop",
        action='store_true',
        help="centercrop",
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
        "--box_descp",
        type=str,
        default='tag',
        help="box_descp",
    )
    parser.add_argument(
        "--spatial_word",
        type=str,
        default=None,
        help="spatial_word",
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
        "--cocogt",
        type=str,
        default="dataset/coco/coco_test30k",
        help="also crop gt coco image",
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
    max_src_length = 77
    if opt.spatial_word is not None:
        # opt.box_descp = 'tag'
        opt.box_descp = 'caption'
    if opt.config != 'configs/stable-diffusion/v1-inference.yaml':
        config.model.params.cond_stage_config.params['extend_outputlen']=616
        config.model.params.cond_stage_config.params['max_length']=616
        max_src_length = 616
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


    ## box processor prepare
    cliptext_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    box_offset = len(cliptext_tokenizer)
    coco_hw = json.load(open('dataset/coco_wh.json', 'r'))
    coco_od = json.load(open('dataset/coco_allbox_gitcaption.json', 'r')) ## 'tag' 'pad_caption' 'crop_caption'
    num_bins, centercrop = 1000, opt.centercrop
    num_withbox = 0
    for key in coco_od:
        if 'box' in coco_od[key]:
            if len(coco_od[key]['box'])!=0: num_withbox+=1
    ## end of box processor prepare

    image_paths, data = [], []
    # meta = json.load(open('dataset/PaintSkills/metadata.json','r'))['Shape']
    meta_color = json.load(open('dataset/PaintSkills/metadata.json','r'))['Color']
    ## ['human', 'airplane', 'bike', 'bus', 'dog', 'boat', 'van', 'train', 'fireHydrant', 'stopSign', 'backpack', 'chair', 'diningTable', 'skateboard', 'bench', 'suitcase', 'trafficLight', 'bird', 'bear', 'bed', 'pottedPlant']
    meta = {0:'person',1:'airplane',2:'bicycle',3:'bus',4:'dog',5:'boat',6:'truck',7:'train',8:'fire hydrant',9:'stop sign',10:'backpack',11:'chair',12:'dining table',13:'skateboard',14:'bench',15:'suitcase',16:'traffic light',17:'bird',18:'bear',19:'bed',20:'potted plant'}

    skills = ['count', 'object', 'spatial', 'color']
    for skill in skills:
        annobox_name = 'dataset/PaintSkills/%s/%s_val_bounding_boxes.json'%(skill,skill)
        anno_box = json.load(open(annobox_name,'r'))
        img_dict = {}
        for annoimgii in anno_box['images']: ## {'file_name': 'image_count_val_00937.png', 'height': 720, 'width': 720, 'id': 937}
            img_dict[annoimgii['id']] = [(annoimgii['height'], annoimgii['width']), []]
        for annoboxii in anno_box['annotations']:  ## {'category_id': 13, 'image_id': 937, 'id': 0, 'shape': 'skateboard', 'state': None, 'iscrowd': 0, 'all_points_x': [], 'all_points_y': [], 'color': 'plain', 'texture': 'plain', 'bbox': [312, 330, 107, 75], 'area': 8025}
            img_dict[annoboxii['image_id']][1].append([annoboxii['id'], annoboxii['bbox'], annoboxii['category_id'], annoboxii['color']])

        anno_name = 'dataset/PaintSkills/%s/scenes/%s_val.json'%(skill,skill)
        anno = json.load(open(anno_name,'r'))
        for annoii in anno['data']: ## {'scene': 'empty', 'skill': 'count', 'objects': [{'id': 0, 'shape': 'humanSophie', 'color': 'plain', 'relation': None, 'scale': 8.890905938624922, 'texture': 'plain', 'rotation': None, 'state': 'sitting'}], 'text': 'a photo of 1 human', 'id': 'count_val_00000'}
            # data.append([annoii['id'],annoii['text'].replace('/',' '),annoii['objects']])
            # data.append([annoii['id'],annoii['text'].replace('/',' '),img_dict[int(annoii['id'][-5:])]])
            data.append([annoii['id'],annoii['text'].replace('/',' ').replace('van','truck').replace('1','one').replace('2','two').replace('3','three').replace('4','four'),\
                img_dict[int(annoii['id'][-5:])]])

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
                        # for prompts in tqdm(data, desc="data"):
                        for sample in tqdm(data, desc="data"):
                            tokenized_text = []
                            str_caption = sample[1]
                            sample_idx = sample[0]
                            boxes = sample[2][1]
                            imagehw = sample[2][0]
                            text = [pre_caption(sample[1].lower(), max_src_length)]
                            text_enc = cliptext_tokenizer(text, truncation=True, max_length=max_src_length, return_length=True,
                                                        return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                            tokenized_text.append(text_enc[0,:])

                            box_list = []
                            for box_info in boxes:
                                box, tag = box_info[1], meta[int(box_info[2])]
                                if skill=='color':
                                    # color = meta_color[int(box_info[-1])]
                                    color = box_info[-1]
                                else:
                                    color = None
                                box[2] = box[0]+box[2]
                                box[3] = box[1]+box[3]
                                if opt.box_descp=='tag':
                                    caption = tag
                                elif 'caption' in opt.box_descp:
                                    caption = 'a photo of a %s'%tag if skill!='color' else 'a photo of a %s %s'%(color, tag)
                                else:
                                    if random.random()<0.5:
                                        caption = tag
                                    else:
                                        caption = 'a photo of a %s'%tag if skill!='color' else 'a photo of a %s %s'%(color, tag)
                                box_list.append((box,caption,tag,color))

                            if len(box_list)!=0:
                                for box_sample in box_list:
                                    if opt.spatial_word is None:
                                        box, caption = box_sample[0], box_sample[1]
                                        box = process_centercrop(box, imagehw, centercrop)
                                        if box[0]==1.0 or box[1]==1.0 or box[2]==0.0 or box[3]==0.0:
                                            continue
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
                                        ## used for naming only, always use tag
                                        str_caption += ' <%d> <%d> <%d> <%d> '%(quant_x0-box_offset,quant_y0-box_offset,quant_x1-box_offset,quant_y1-box_offset) + box_sample[2]
                                    else:
                                        box, caption = box_sample[0], box_sample[1]
                                        box = process_centercrop(box, imagehw, centercrop)
                                        if box[0]==1.0 or box[1]==1.0 or box[2]==0.0 or box[3]==0.0:
                                            continue
                                        if box[0]>=box[2] or box[1]>=box[3]:
                                            continue
                                        region_word = process_spatial_word(box, caption, opt.spatial_word)
                                        region_word = pre_caption(region_word.lower(), max_src_length)
                                        region_text = cliptext_tokenizer(region_word, truncation=True, max_length=max_src_length, return_length=True,
                                                                    return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                                        tokenized_text.append(region_text[0,:])
                                        str_caption += region_word

                            if opt.config == 'configs/stable-diffusion/v1-inference.yaml' and opt.spatial_word is None:
                                tokenized_text = tokenized_text[0][:max_src_length]
                            else:
                                tokenized_text = torch.cat(tokenized_text, dim=0)[:max_src_length]


                            pad_tokenized_text = torch.tensor([box_offset-1]*max_src_length).to(text_enc.device)
                            pad_tokenized_text[:len(tokenized_text)] = tokenized_text
                            prompts = pad_tokenized_text.unsqueeze(0)
                            opt.prompt, prompt = str_caption[:200], str_caption[:200]
                            # example["caption"] = pad_tokenized_text
                            # example["str_caption"] = str_caption


                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])

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
                                    while os.path.isfile(os.path.join(outpath, "%s/%s_%d.png"%(skill,prompt,repeat))):
                                        repeat+=1
                                    pred_image = Image.fromarray(x_sample.astype(np.uint8))
                                    pred_image.save(os.path.join(outpath, "%s/%s_%d.png"%(skill,prompt,repeat)))
                                    base_count += 1


                            for ii in range(len(box_list)):
                                os.makedirs(os.path.join(outpath,'%s_crop'%skill), exist_ok=True)
                                os.makedirs(os.path.join(outpath,'%s_crop_foldered'%skill), exist_ok=True)
                                box = box_list[ii][0]
                                pred_dim = pred_image.size[0]
                                left, top, right, bottom = int(box[0]*pred_dim), int(box[1]*pred_dim), int(box[2]*pred_dim), int(box[3]*pred_dim)
                                left, top, right, bottom = min(max(0,left),pred_dim), min(max(0,top),pred_dim), min(max(0,right),pred_dim), min(max(0,bottom),pred_dim)
                                if left>=right or top>=bottom:
                                    continue
                                pred_crop = pred_image.crop((left, top, right, bottom))
                                pred_crop.save(os.path.join(outpath, '%s_crop/%s_%d_%d.png'%(skill,prompt,repeat,ii)))
                                ## save for CLS
                                category = box_list[ii][2]
                                os.makedirs(os.path.join(outpath, '%s_crop_foldered/%s'%(skill,category)), exist_ok=True)
                                pred_crop.save(os.path.join(outpath, '%s_crop_foldered/%s/%s_%d_%d.png'%(skill,category,prompt,repeat,ii)))                    
                                ## save for CLS
                                if skill=='color':
                                    os.makedirs(os.path.join(outpath,'%s_crop_attr_foldered'%skill), exist_ok=True)
                                    attribute = box_list[ii][3]
                                    os.makedirs(os.path.join(outpath,'%s_crop_attr_foldered/%s'%(skill,attribute)), exist_ok=True)
                                    pred_crop.save(os.path.join(outpath,'%s_crop_attr_foldered/%s/%s_%d_%d.png'%(skill,attribute,prompt,repeat,ii)))                    

                    toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
