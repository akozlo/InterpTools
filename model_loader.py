import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache
import hashlib

MODEL_OPTIONS = {
    "gpt2-small": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "Qwen 2-0.5B": "Qwen/Qwen2-0.5B",
    "Qwen 2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen 2-1.5B": "Qwen/Qwen2-1.5B",
    "Qwen 2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    # Add more models here as they become available
}

def load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a pretrained model using TransformerLens.
    
    Args:
    model_name (str): The friendly name of the model
    device (str): The device to load the model on (default: "cuda" if available, else "cpu")

    Returns:
    HookedTransformer: The loaded model
    """
    from transformer_lens import HookedTransformer

    model_path = get_model_path(model_name)
    if not model_path:
        raise ValueError(f"Model '{model_name}' not found in the registry.")

    print(f"Loading model {model_name} from {model_path}...")
    model = HookedTransformer.from_pretrained(model_path, device=device)
    print(f"Model loaded successfully on {device}.")

    return model


def get_model_path(model_name):
    """
    Get the Hugging Face model path for a given model name.
    
    Args:
    model_name (str): The friendly name of the model

    Returns:
    str: The Hugging Face model path, or None if not found
    """
    return MODEL_OPTIONS.get(model_name)

def list_available_models():
    """
    Return a list of available models.

    Returns:
    list: A list of model names.
    """
    return list(MODEL_OPTIONS.keys())

# You can add more utility functions here as needed


def print_available_models():
    """
    Print all available models in the registry.
    """
    models = list_available_models()
    print("Available models:")
    for model in models:
        print(f"- {model}")

def gpu_mem_check():
    """
    Check and print the current GPU memory usage.
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # Convert to GB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
        cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB

        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
        print(f"Cached GPU Memory: {cached_memory:.2f} GB")
    else:
        print("CUDA is not available. Please check your GPU setup.")


# Example usage in notebook:
# print_available_models()
# model, tokenizer = load_model('Llama 3.1-8B')
# model, tokenizer = load_model('meta-llama/Meta-Llama-3.1-8B', quantization='8bit')  # Using full path
# model, tokenizer = load_model('Qwen 2-1.5B-Instruct', custom_quantized_model_id='Qwen/Qwen2-1.5B-Instruct-8bit')
# gpu_mem_check()  # Call this function to check GPU memory
