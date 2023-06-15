import os,math,re,json
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random
from transformers import CLIPTokenizer, CLIPTextModel
import ldm.data.transforms as T

# from ldm.data.file_dataset import FileDataset
import re

class PersonalizedBase_Text(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg = False,
                 coco_path = 'dataset/coco/all2014',
                 with_bbox = False,
                 num_bins = 1000,
                 max_src_length = 77,
                 box_descp = 'tag',
                 spatial_word = None,
                 ):

        self.data_root = data_root
        self.coco_path = coco_path
        self.max_src_length = max_src_length

        self.image_paths = []
        if 'tsv' in self.data_root:
            lines = [line.rstrip() for line in open(self.data_root)]
            for ii in range(len(lines)):
                value = re.split(r'\t', lines[ii])
                self.image_paths.append([value[0],value[-1].replace('/',' ')])
        else:
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        print('number of samples:',self._length,data_root,set)
        self.with_bbox = with_bbox
        self.num_bins = num_bins
        self.coco_hw = json.load(open('dataset/coco_wh.json', 'r'))
        if self.with_bbox:
            self.box_descp = box_descp
            self.spatial_word = spatial_word
            if self.spatial_word is not None:
                # self.box_descp = 'tag'
                self.box_descp = 'caption'
            self.cliptext_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.box_offset = len(self.cliptext_tokenizer)
            self.coco_od = json.load(open('dataset/coco_allbox_gitcaption.json', 'r')) ## 'tag' 'pad_caption' 'crop_caption'  ## GIT-COCO ckpt
            # self.coco_od = json.load(open('dataset/coco_allbox_gitcaption_coco.json', 'r')) ## GIT-COCO ckpt
            num_withbox = 0
            for key in self.coco_od:
                if 'box' in self.coco_od[key]:
                    if len(self.coco_od[key]['box'])!=0: num_withbox+=1
            print('images with box',num_withbox,'images',len(self.coco_od),'dataset len',self._length)

        ##
        print('always using Image.BICUBIC in resizing.')
        if set == "train":
            self._transforms = T.Compose(
                [ 
                    T.RandomHorizontalFlip(p=flip_p),
                    T.RandomResize([self.size]),
                    T.RandomCrop((self.size,self.size)), 
                ]
            )
        else:
            self._transforms = T.Compose(
                [
                    T.RandomResize([self.size]),
                    T.CenterCrop((self.size,self.size)),
                ]
            )            


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        sample = self.image_paths[i % self.num_images] ## [cocoid, caption]
        sample_idx = sample[0]
        img_path = self.coco_path + '/COCO_train2014_%012d.jpg'%(int(sample_idx))
        if not os.path.isfile(img_path):
            img_path = self.coco_path + '/COCO_val2014_%012d.jpg'%(int(sample_idx))

        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        imagehw = self.coco_hw[sample_idx]
        target = {"image_id": sample_idx, "annotations": sample_idx, "caption": sample[1], \
                    "orig_size": torch.tensor([imagehw[0], imagehw[1]]), "size": torch.tensor([imagehw[0], imagehw[1]])}

        if self.with_bbox:
            box_list, regioncap_list = [], []
            sample_p = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.] ## max
            if sample_idx in self.coco_od:
                if 'box' in self.coco_od[sample_idx]:   ## Use OD box instead
                    od_anno = self.coco_od[sample_idx]
                    number_box = random.choices(list(range(len(sample_p))), weights=sample_p)[0]
                    number_box = min(number_box, len(od_anno['box']))
                    # number_box = len(od_anno['box'])
                    # ious = [x['iou'] for x in od_anno['box']]   ## object sample based on size
                    ious = [math.sqrt(x['iou']) for x in od_anno['box']]   ## object sample based on size
                    norm_ious = [x/sum(ious) for x in ious]
                    boxes = od_anno['box']

                    for sample_ii in range(number_box):
                        idx = random.choices(list(range(len(ious))), weights=norm_ious)[0]
                        box_sample = boxes.pop(idx)
                        iou_sample = ious.pop(idx)
                        normiou_sample = norm_ious.pop(idx)
                        box = box_sample['bbox']
                        box[2] = box[0]+box[2]
                        box[3] = box[1]+box[3]
                        if self.box_descp=='tag':
                            caption = box_sample['tag']
                        else:
                            caption = box_sample['crop_caption'] if '"' not in box_sample['crop_caption'] else box_sample['pad_caption']
                        box_list.append(box)
                        regioncap_list.append(caption)
                target["boxes"] = torch.as_tensor(box_list, dtype=torch.float32).reshape(-1, 4)
                target["box_caption"] = regioncap_list
        image, target = self._transforms(image, target)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        if not self.with_bbox:
            example["caption"] = target["caption"]
            example["str_caption"] = target["caption"]
        else:
            tokenized_text = []
            str_caption = target["caption"]
            text = [self.pre_caption(target["caption"].lower(), self.max_src_length)]
            text_enc = self.cliptext_tokenizer(text, truncation=True, max_length=self.max_src_length, return_length=True,
                                        return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
            tokenized_text.append(text_enc[0,:])

            if len(box_list)!=0:
                imagehw = self.coco_hw[sample_idx]
                # for box_sample_ii in range(len(box_list)):
                for box_sample_ii in range(target["boxes"].shape[0]):
                    if self.spatial_word is None:
                        box, caption = target["boxes"][box_sample_ii], target["box_caption"][box_sample_ii]
                        # box = self.process_centercrop(box, imagehw, self.center_crop)
                        box = [float(x)/self.size for x in box]
                        quant_x0 = int(round((box[0] * (self.num_bins - 1)))) + self.box_offset
                        quant_y0 = int(round((box[1] * (self.num_bins - 1)))) + self.box_offset
                        quant_x1 = int(round((box[2] * (self.num_bins - 1)))) + self.box_offset
                        quant_y1 = int(round((box[3] * (self.num_bins - 1)))) + self.box_offset
                        region_coord = torch.tensor([quant_x0,quant_y0,quant_x1,quant_y1]).to(text_enc.device)
                        caption = self.pre_caption(caption.lower(), self.max_src_length)
                        region_text = self.cliptext_tokenizer(caption, truncation=True, max_length=self.max_src_length, return_length=True,
                                                    return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                        tokenized_text.append(region_coord)
                        tokenized_text.append(region_text[0,:])
                        str_caption += ' <%d> <%d> <%d> <%d> '%(quant_x0-self.box_offset,quant_y0-self.box_offset,quant_x1-self.box_offset,quant_y1-self.box_offset) + caption
                    else:
                        box, caption = target["boxes"][box_sample_ii], target["box_caption"][box_sample_ii]
                        box = [float(x)/self.size for x in box]
                        region_word = self.process_spatial_word(box, caption, self.spatial_word)
                        region_word = self.pre_caption(region_word.lower(), self.max_src_length)
                        region_text = self.cliptext_tokenizer(region_word, truncation=True, max_length=self.max_src_length, return_length=True,
                                                    return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                        tokenized_text.append(region_text[0,:])
                        str_caption += region_word

            tokenized_text = torch.cat(tokenized_text, dim=0)[:self.max_src_length]
            pad_tokenized_text = torch.tensor([self.box_offset-1]*self.max_src_length).to(text_enc.device)
            pad_tokenized_text[:len(tokenized_text)] = tokenized_text
            example["caption"] = pad_tokenized_text
            example["str_caption"] = str_caption
        return example

    #####################################
    ## 32*32 / (640*480) = 0.003333333
    ## 96*96 / (640*480) = 0.03
    def process_spatial_word(self, box, caption, spatial_word_mode='all'):
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

    #####################################
    def process_centercrop(self, boxes, imagehw, centercrop):
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

    def pre_caption(self, caption, max_words):
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
