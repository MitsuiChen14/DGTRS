"""
Convert DGTRS-CLIP .pt checkpoint to HuggingFace Diffusers format.

This utility converts the PyTorch checkpoint format to HuggingFace Diffusers
compatible format, allowing the model to be used with diffusers library
for Stable Diffusion and other diffusion models.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_text_encoder_config(output_dir: Path, model_config: Dict[str, Any]):
    """Create config.json for the text encoder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    text_config = model_config["text_config"]
    config = {
        "architectures": ["CLIPTextModel"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": text_config["hidden_size"],
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": text_config["intermediate_size"],
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": text_config["max_position_embeddings"],
        "model_type": "clip_text_model",
        "num_attention_heads": text_config["num_attention_heads"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "pad_token_id": 1,
        "projection_dim": model_config["projection_dim"],
        "torch_dtype": "float32",
        "transformers_version": "4.25.0.dev0", # Using a common version
        "vocab_size": text_config["vocab_size"]
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Saved text_encoder config.json to {config_path}")


def extract_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Extract model configuration from state dict."""
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
    }
    
    model_config = {
        "text_config": text_config,
        "projection_dim": embed_dim,
    }
    
    return model_config


def convert_state_dict_to_diffusers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert state dict to HuggingFace Diffusers text_encoder format."""
    hf_state_dict = {}
    
    for key, value in state_dict.items():
        if not (key.startswith("transformer.") or key in ["token_embedding.weight", "positional_embedding", "ln_final.weight", "ln_final.bias"]):
            continue

        new_key = key
        if key.startswith("transformer."):
            new_key = "text_model." + new_key
            new_key = new_key.replace("transformer.resblocks", "encoder.layers")
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
        
        elif key == "token_embedding.weight":
            new_key = "text_model.embeddings.token_embedding.weight"
        elif key == "positional_embedding":
            new_key = "text_model.embeddings.position_embedding.weight"
        elif key.startswith("ln_final"):
            new_key = key.replace("ln_final", "text_model.final_layer_norm")
        
        hf_state_dict[new_key] = value

    return hf_state_dict


def convert_to_diffusers(
    checkpoint_path: str,
    output_dir: str,
):
    """
    Convert DGTRS-CLIP .pt checkpoint to a Diffusers-compatible text_encoder.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save the converted text_encoder
    
    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    print("Extracting model configuration...")
    model_config = extract_model_config(state_dict)
    
    print("Converting state dict to Diffusers format...")
    hf_state_dict = convert_state_dict_to_diffusers(state_dict)
    
    # Diffusers expects the text_encoder to be in a subdirectory
    text_encoder_dir = output_dir / "text_encoder"
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = text_encoder_dir / "pytorch_model.bin"
    torch.save(hf_state_dict, model_path)
    print(f"Saved model to {model_path}")
    
    # Save config file
    create_text_encoder_config(text_encoder_dir, model_config)

    print(f"\nConversion complete! Text encoder saved to {text_encoder_dir}")
    print(f"\nTo use with diffusers, point to the parent directory:")
    print(f"  from transformers import CLIPTextModel")
    print(f"  text_encoder = CLIPTextModel.from_pretrained('{output_dir}/text_encoder')")
    
    return str(output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert DGTRS-CLIP checkpoint to HuggingFace Diffusers format")
    parser.add_argument("checkpoint_path", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("output_dir", type=str, help="Output directory for converted model")
    
    args = parser.parse_args()
    
    convert_to_diffusers(
        args.checkpoint_path,
        args.output_dir,
    )

