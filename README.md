# ReCo: Region-Controlled Text-to-Image Generation
[ReCo: Region-Controlled Text-to-Image Generation](https://arxiv.org/pdf/2211.15518.pdf)

by [Zhengyuan Yang](https://zhengyuan.info), [Jianfeng Wang](http://jianfengwang.me/), [Zhe Gan](https://zhegan27.github.io/), [Linjie Li](https://www.microsoft.com/en-us/research/people/linjli/), [Kevin Lin](https://sites.google.com/site/kevinlin311tw/), [Chenfei Wu](https://chenfei-wu.github.io/), [Nan Duan](https://nanduan.github.io/), [Zicheng Liu](https://zicliu.wixsite.com/mysite), [Ce Liu](http://people.csail.mit.edu/celiu/), [Michael Zeng](https://www.microsoft.com/en-us/research/people/nzeng/), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/)

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

**\*\*\*\*\* Update: ReCo is now available at Huggingface Diffusers! [COCO Model](https://huggingface.co/j-min/reco_sd14_coco) [LAION Model](https://huggingface.co/j-min/reco_sd14_laion). 

Credits to [Jaemin Cho](https://j-min.io/). Thank you! \*\*\*\*\***


### Introduction
ReCo extends T2I models to understand coordinate inputs. Thanks to the introduced position tokens in the region-controlled input query, users can easily specify free-form regional descriptions in arbitrary image regions.
For more details, please refer to our
[paper](https://arxiv.org/pdf/2211.15518.pdf).


<p align="center">
  <img src="https://zyang-ur.github.io//reco/reco.png" width="100%"/>
</p>

### Citation

    @inproceedings{yang2023reco,
      title={ReCo: Region-Controlled Text-to-Image Generation},
      author={Yang, Zhengyuan and Wang, Jianfeng and Gan, Zhe and Li, Linjie and Lin, Kevin and Wu, Chenfei and Duan, Nan and Liu, Zicheng and Liu, Ce and Zeng, Michael and Wang, Lijuan},
      booktitle={CVPR},
      year={2023}
    }


## Installation
Clone the repository:
```
git clone https://github.com/microsoft/ReCo.git
cd ReCo
```

A [conda](https://conda.io/) environment named `reco_env` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate reco_env
```

Or install packages in ``requirements.txt``:

```
pip install -r requirements.txt
```

### AzCopy
We recommend using the following AzCopy command to download.
AzCopy executable tools can be [downloaded here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

Example command:
```
path/to/azcopy copy <folder-link> <target-address> --resursive"

# For example:
path/to/azcopy copy https://unitab.blob.core.windows.net/data/reco/dataset <local_path> --recursive
```

## Data
Download processed dataset annotations ```dataset``` folder in the following dataset path (~59G) with [azcopy tool](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).
```
path/to/azcopy copy https://unitab.blob.core.windows.net/data/reco/dataset <local_path> --recursive
```


## Inference and Checkpoints
ReCo checkpoints trained on COCO and a small LAION subset can be downloaded via wget or [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) here [ReCo_COCO](https://unitab.blob.core.windows.net/data/reco/reco_coco_616.ckpt) and [ReCo_LAION](https://unitab.blob.core.windows.net/data/reco/reco_laion_1232.ckpt). Save downloaded weights to ```logs```.

```inference.sh``` contains examples for inference calls

```eval.sh``` contains examples for coco evaluation.

## Fine-tuning
For ReCo fine-tuning, we start with the stable diffusion model with [instructions here](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1). Weights can be downloaded on [HuggingFace](https://huggingface.co/CompVis). The experiments mainly use ```sd-v1-4-full-ema.ckpt```.

```train.sh``` contains examples for fine-tuning.


## Acknowledgement
The project is built based on the following repository:
* [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion),
* [XavierXiao/Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion),
* [harubaru/waifu-diffusion: stable diffusion finetuned on danbooru](https://github.com/harubaru/waifu-diffusion).

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.