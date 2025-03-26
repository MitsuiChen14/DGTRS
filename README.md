# LRSCLIP 🚀

This is the open-source repository for the paper 《**LRSCLIP: A Vision-Language Foundation Model for Aligning Remote Sensing Images with Longer Text**》.

Weizhi Chen, Jingbo Chen, Yupeng Deng*, Jiansheng Chen*, Yuman Feng, Zhihao Xi, Diyou Liu, Kai Li, Yu Meng

## Highlights ✨

* 🔥LRSCLIP is the first vision-language foundation model in the remote sensing domain that can simultaneously handle both long and short texts.
* 🔥LRS2M is the first image-text dataset in the remote sensingdomain with 2 million data points that simultaneously provides both long and short text captions for images.
* 🔥LRSCLIP achieves SOTA performance in four zero-shot downstream tasks: long-text cross-modal retrieval, short-text cross-modal retrieval, image classification, and semantic localization.

## Usage 💻

### Installation

Our model is based on [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), please prepare environment for CLIP.

### how to sue

Please first clone our [repo](https://github.com/MitsuiChen14/LRSCLIP) from github by running the following command.

```shell
git clone https://github.com/MitsuiChen14/LRSCLIP.git
cd LRSCLIP
```

Then, download the [LRSCLIP-B/16](https://huggingface.co/cwz14/LRSCLIP_ViT-B-16) checkpoint and place them in the `./checkpoints` directory.

### Demo
```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

text = longclip.tokenize(["Numerous students are walking in the green pass in this campus.", "These buildings belong to the school buildings.", "There are many residential areas near the school."]).to(device)
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
```


## Todo 📝
- [x] Release the [paper](http://arxiv.org/abs/2503.19311) on arXiv.
- [x] LRSCLIP's checkpoints([LRSCLIP_ViT-B-16](https://huggingface.co/cwz14/LRSCLIP_ViT-B-16) and [LRSCLIP_ViT-L-14](https://huggingface.co/cwz14/LRSCLIP_ViT-L-14)) and usage demo.
- [ ] The evaluation code for the LRSCLIP.

## Citation 📖

If you find our work helpful for your research, please consider giving a citation:

```
@article{chen2025lrsclipvisionlanguagefoundationmodel,
      title={LRSCLIP: A Vision-Language Foundation Model for Aligning Remote Sensing Image with Longer Text}, 
      author={Weizhi Chen and Jingbo Chen and Yupeng Deng and Jiansheng Chen and Yuman Feng and Zhihao Xi and Diyou Liu and Kai Li and Yu Meng},
      year={2025},
      eprint={2503.19311},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19311}, 
      journal={ArXiv},
      volume={abs/2503.19311},
}
```









