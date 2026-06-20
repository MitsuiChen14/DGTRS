# DGTRS-CLIP Model Conversion Utilities

This directory contains utilities to convert DGTRS-CLIP PyTorch checkpoints (`.pt` files) to HuggingFace formats for use with the `transformers` and `diffusers` libraries.

## Available Utilities

### 1. Convert to HuggingFace Transformers Format

Converts `.pt` checkpoints to the standard HuggingFace Transformers format for `CLIPModel`, following the structure of models like `openai/clip-vit-base-patch32`.

**Usage:**
```python
from utils import convert_to_transformers

convert_to_transformers(
    checkpoint_path="./checkpoints/DGTRS-CLIP-ViT-B-16/DGTRS-CLIP-ViT-B-16.pt",
    output_dir="./checkpoints/DGTRS-CLIP-ViT-B-16_transformers_model"
)
```

**Command line:**
```bash
python -m utils.convert_to_transformers \
    ./checkpoints/DGTRS-CLIP-ViT-B-16/DGTRS-CLIP-ViT-B-16.pt \
    ./checkpoints/DGTRS-CLIP-ViT-B-16_transformers_model
```

**Output structure:**
```
./checkpoints/DGTRS-CLIP-ViT-B-16_transformers_model/
├── config.json
├── merges.txt
├── preprocessor_config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.json
```

### 2. Convert to HuggingFace Diffusers Format

Converts the text encoder from a `.pt` checkpoint to the format required by `diffusers` pipelines like Stable Diffusion.

**Usage:**
```python
from utils import convert_to_diffusers

convert_to_diffusers(
    checkpoint_path="./checkpoints/DGTRS-CLIP-ViT-B-16/DGTRS-CLIP-ViT-B-16.pt",
    output_dir="./checkpoints/DGTRS-CLIP-ViT-B-16_diffusers_model"
)
```

**Command line:**
```bash
python -m utils.convert_to_diffusers \
    ./checkpoints/DGTRS-CLIP-ViT-B-16/DGTRS-CLIP-ViT-B-16.pt \
    ./checkpoints/DGTRS-CLIP-ViT-B-16_diffusers_model
```

**Output structure:**
```
./checkpoints/DGTRS-CLIP-ViT-B-16_diffusers_model/
└── text_encoder/
    ├── config.json
    └── pytorch_model.bin
```

## Using Converted Models

### With Transformers

```python
from transformers import CLIPModel, CLIPProcessor

# Load the converted model
model = CLIPModel.from_pretrained("./checkpoints/DGTRS-CLIP-ViT-B-16_transformers_model")
processor = CLIPProcessor.from_pretrained("./checkpoints/DGTRS-CLIP-ViT-B-16_transformers_model")

# Use the model for image-text similarity
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], 
#                    images=images, return_tensors="pt", padding=True)
# outputs = model(**inputs)
```

### With Diffusers

```python
from transformers import CLIPTextModel

# Load the converted text encoder
text_encoder = CLIPTextModel.from_pretrained("./checkpoints/DGTRS-CLIP-ViT-B-16_diffusers_model/text_encoder")

# Use the text encoder to get text embeddings
# text_inputs = tokenizer(...)
# text_embeddings = text_encoder(**text_inputs).last_hidden_state
```

## Notes

- `convert_to_transformers` converts the full CLIP model (vision and text encoders).
- `convert_to_diffusers` converts only the text encoder for use in diffusion model pipelines.
- The conversion preserves all relevant model weights.
- All necessary configuration and tokenizer files are generated for seamless integration.
- The converted models maintain the same functionality as the original `.pt` checkpoint.

## Requirements

- PyTorch
- transformers
- diffusers

