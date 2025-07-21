# DGTRS 🚀

This is the open-source repository for the paper 《**DGTRSD & DGTRS-CLIP: A Dual-Granularity Remote Sensing Image–Text Dataset and Vision Language Foundation Model for Alignment**》.

Weizhi Chen, Jin Wei, Yupeng Deng*, Jingbo Chen, Jiansheng Chen*, Yuman Feng, Zhihao Xi, Diyou Liu, Kai Li, Yu Meng

## ✨ Highlights ✨

* 🔥DGTRS-CLIP is the first vision-language foundation model in the remote sensing domain that can simultaneously handle both long and short texts.
* 🔥DGTRSD is the first image-text dataset in the remote sensingdomain with 2 million data points that simultaneously provides both long and short text captions for images.
* 🔥DGTRS-CLIP achieves SOTA performance in four zero-shot downstream tasks: long-text cross-modal retrieval, short-text cross-modal retrieval, image classification, and semantic localization.

## 💻 Usage 💻

### Installation

Our model is based on [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), please prepare environment for CLIP.

### How to sue

Please first clone our [repo](https://github.com/MitsuiChen14/DGTRS.git) from github by running the following command.

```shell
git clone https://github.com/MitsuiChen14/DGTRS.git
cd DGTRS
```

Then, download the [DGTRS-CLIP-ViT-B-16](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-B-16)checkpoint and place them in the `./checkpoints` directory.

### Demo
```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/DGTRS-CLIP-ViT-B-16, device=device)

text = longclip.tokenize(["Numerous students are walking in the green pass in this campus.", "These buildings belong to the school buildings.", "There are many residential areas near the school."]).to(device)
image = preprocess(Image.open("./img/demo.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
```


## 📝 Todo 📝
- [x] Release the [paper](http://arxiv.org/abs/2503.19311) on arXiv.
- [x] LRSCLIP's checkpoints([DGTRS-CLIP-ViT-B-16](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-B-16) and [DGTRS-CLIP-ViT-L-14](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-L-14)) and usage demo.
- [ ] The training dataset DGTRSD for DGTRS-CLIP.
- [ ] The evaluation code for the DGTRS-CLIP.

## 📖 Citation 📖

If you find our work helpful for your research, please consider giving a citation:

```
@article{chen2025lrsclip,
  title={LRSCLIP: A Vision-Language Foundation Model for Aligning Remote Sensing Image with Longer Text},
  author={Chen, Weizhi and Chen, Jingbo and Deng, Yupeng and Chen, Jiansheng and Feng, Yuman and Xi, Zhihao and Liu, Diyou and Li, Kai and Meng, Yu},
  journal={arXiv preprint arXiv:2503.19311},
  year={2025}
}
```









