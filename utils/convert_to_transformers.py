"""
Convert DGTRS-CLIP .pt checkpoint to HuggingFace Transformers format.

This utility converts the PyTorch checkpoint format to HuggingFace Transformers
compatible format, allowing the model to be used with transformers library.
"""

import os
import json
import torch
import gzip
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.simple_tokenizer import SimpleTokenizer, bytes_to_unicode


def create_tokenizer_files(output_dir: Path):
    """Create vocab.json and merges.txt for the CLIP tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = SimpleTokenizer()
    
    # Save vocab.json
    vocab_json_path = output_dir / "vocab.json"
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.encoder, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab.json to {vocab_json_path}")

    # Save merges.txt from the gzipped bpe file
    bpe_path = Path(__file__).parent.parent / "model" / "bpe_simple_vocab_16e6.txt.gz"
    merges_txt_path = output_dir / "merges.txt"
    with gzip.open(bpe_path, 'rt', encoding='utf-8') as f_in:
        with open(merges_txt_path, 'w', encoding='utf-8') as f_out:
            # Skip the version line
            next(f_in)
            f_out.write(f_in.read())
    print(f"Saved merges.txt to {merges_txt_path}")


def create_config_files(output_dir: Path, model_config: Dict[str, Any]):
    """Create config.json, tokenizer_config.json, and preprocessor_config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config.json
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    print(f"Saved config.json to {config_path}")

    # Save tokenizer_config.json
    tokenizer_config = {
        "model_max_length": model_config["text_config"]["max_position_embeddings"],
        "padding_side": "right",
        "tokenizer_class": "CLIPTokenizer",
    }
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Saved tokenizer_config.json to {tokenizer_config_path}")
    
    # Save special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    special_tokens_map_path = output_dir / "special_tokens_map.json"
    with open(special_tokens_map_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"Saved special_tokens_map.json to {special_tokens_map_path}")

    # Save preprocessor_config.json
    preprocessor_config = {
        "crop_size": {
            "height": model_config["vision_config"]["image_size"],
            "width": model_config["vision_config"]["image_size"]
        },
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_resize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3, # BICUBIC
        "size": {
            "shortest_edge": model_config["vision_config"]["image_size"]
        }
    }
    preprocessor_config_path = output_dir / "preprocessor_config.json"
    with open(preprocessor_config_path, "w") as f:
        json.dump(preprocessor_config, f, indent=2)
    print(f"Saved preprocessor_config.json to {preprocessor_config_path}")


def extract_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Extract model configuration from state dict."""
    config = {}
    
    vit = "visual.proj" in state_dict
    
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        
        vision_config = {
            "hidden_size": vision_width,
            "intermediate_size": vision_width * 4,
            "num_attention_heads": vision_width // 64,
            "num_hidden_layers": vision_layers,
            "patch_size": vision_patch_size,
            "image_size": image_resolution,
            "projection_dim": state_dict["text_projection"].shape[1]
        }
    else:
        raise NotImplementedError("ResNet conversion not implemented in this version.")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    text_config = {
        "hidden_size": transformer_width,
        "intermediate_size": transformer_width * 4,
        "max_position_embeddings": context_length,
        "num_attention_heads": transformer_heads,
        "num_hidden_layers": transformer_layers,
        "vocab_size": vocab_size,
        "projection_dim": embed_dim
    }
    
    model_config = {
        "model_type": "clip",
        "text_config": text_config,
        "vision_config": vision_config,
        "projection_dim": embed_dim,
        "logit_scale_init_value": 2.6592
    }
    
    return model_config


def convert_state_dict_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert state dict from original format to HuggingFace Transformers format."""
    hf_state_dict = {}
    for key, value in state_dict.items():
        if key == "visual.proj":
            hf_state_dict["visual_projection.weight"] = value
            continue
        if key == "text_projection":
            hf_state_dict["text_projection.weight"] = value
            continue

        if key.startswith("visual."):
            new_key = key.replace("visual.", "vision_model.")
            if "conv1" in new_key:
                new_key = new_key.replace("conv1.weight", "embeddings.patch_embedding.weight")
            elif "class_embedding" in new_key:
                 new_key = new_key.replace("class_embedding", "embeddings.class_embedding")
            elif "positional_embedding" in new_key:
                 new_key = new_key.replace("positional_embedding", "embeddings.position_embedding.weight")
            elif "ln_pre" in new_key:
                new_key = new_key.replace("ln_pre", "pre_layernorm")
            elif "transformer.resblocks" in new_key:
                new_key = new_key.replace("transformer.resblocks", "encoder.layers")
                new_key = new_key.replace("attn", "self_attn")
                new_key = new_key.replace("in_proj_", "qkv_")
                if "qkv_weight" in new_key:
                    q, k, v = value.chunk(3, dim=0)
                    layer = new_key.split(".")[3]
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.q_proj.weight"] = q
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.k_proj.weight"] = k
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.v_proj.weight"] = v
                    continue
                if "qkv_bias" in new_key:
                    q, k, v = value.chunk(3, dim=0)
                    layer = new_key.split(".")[3]
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.q_proj.bias"] = q
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.k_proj.bias"] = k
                    hf_state_dict[f"vision_model.encoder.layers.{layer}.self_attn.v_proj.bias"] = v
                    continue
                new_key = new_key.replace("out_proj", "out_proj")
                new_key = new_key.replace("ln_1", "layer_norm1")
                new_key = new_key.replace("ln_2", "layer_norm2")
                new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
                new_key = new_key.replace("mlp.c_proj", "mlp.fc2")
            elif "ln_post" in new_key:
                new_key = new_key.replace("ln_post", "post_layernorm")
            hf_state_dict[new_key] = value

        elif key.startswith("transformer."):
            new_key = key.replace("transformer.", "text_model.")
            new_key = new_key.replace("resblocks", "encoder.layers")
            new_key = new_key.replace("attn", "self_attn")
            if "in_proj_weight" in new_key:
                 q, k, v = value.chunk(3, dim=0)
                 layer = new_key.split(".")[3]
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.q_proj.weight"] = q
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.k_proj.weight"] = k
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.v_proj.weight"] = v
                 continue
            if "in_proj_bias" in new_key:
                 q, k, v = value.chunk(3, dim=0)
                 layer = new_key.split(".")[3]
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.q_proj.bias"] = q
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.k_proj.bias"] = k
                 hf_state_dict[f"text_model.encoder.layers.{layer}.self_attn.v_proj.bias"] = v
                 continue
            new_key = new_key.replace("out_proj", "out_proj")
            new_key = new_key.replace("ln_1", "layer_norm1")
            new_key = new_key.replace("ln_2", "layer_norm2")
            new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
            new_key = new_key.replace("mlp.c_proj", "mlp.fc2")
            hf_state_dict[new_key] = value
        
        elif key == "token_embedding.weight":
            hf_state_dict["text_model.embeddings.token_embedding.weight"] = value
        elif key == "positional_embedding":
            hf_state_dict["text_model.embeddings.position_embedding.weight"] = value
        elif key.startswith("ln_final"):
            hf_state_dict[key.replace("ln_final", "text_model.final_layer_norm")] = value
        elif key == "logit_scale":
            hf_state_dict[key] = value

    return hf_state_dict


def convert_to_transformers(
    checkpoint_path: str,
    output_dir: str,
):
    """
    Convert DGTRS-CLIP .pt checkpoint to HuggingFace Transformers format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save the converted model
    
    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    print("Extracting model configuration...")
    model_config = extract_model_config(state_dict)
    
    print("Converting state dict to Transformers format...")
    hf_state_dict = convert_state_dict_to_hf(state_dict)
    
    # Save the model
    model_path = output_dir / "pytorch_model.bin"
    torch.save(hf_state_dict, model_path)
    print(f"Saved model to {model_path}")
    
    # Save config files
    create_config_files(output_dir, model_config)
    
    # Save tokenizer files
    create_tokenizer_files(output_dir)

    print(f"\nConversion complete! Model saved to {output_dir}")
    print(f"\nTo use with transformers:")
    print(f"  from transformers import CLIPModel, CLIPProcessor")
    print(f"  model = CLIPModel.from_pretrained('{output_dir}')")
    print(f"  processor = CLIPProcessor.from_pretrained('{output_dir}')")
    
    return str(output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert DGTRS-CLIP checkpoint to HuggingFace Transformers format")
    parser.add_argument("checkpoint_path", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("output_dir", type=str, help="Output directory for converted model")
    
    args = parser.parse_args()
    
    convert_to_transformers(
        args.checkpoint_path,
        args.output_dir,
    )

