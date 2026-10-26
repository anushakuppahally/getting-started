
# trr_model.py

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts_features import (
    trace_to_features,
    trace_to_series,
)

_CURRENT_MODEL_ID = None
_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None


def load_model(model_id):
    global _CURRENT_MODEL_ID, _CURRENT_MODEL, _CURRENT_TOKENIZER

    if _CURRENT_MODEL_ID == model_id:
        return _CURRENT_TOKENIZER, _CURRENT_MODEL

    if _CURRENT_MODEL is not None:
        del _CURRENT_MODEL, _CURRENT_TOKENIZER
        _CURRENT_MODEL = None
        _CURRENT_TOKENIZER = None
        _CURRENT_MODEL_ID = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device == "cuda" else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    _CURRENT_MODEL_ID = model_id
    _CURRENT_MODEL = model
    _CURRENT_TOKENIZER = tokenizer

    return tokenizer, model


def make_prompt(tokenizer, problem):
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following mathematical problem. "
                "Reason step by step, then give your final answer.\n\n"
                + problem
            ),
        }
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return messages[0]["content"]


@torch.inference_mode()
def get_hidden_states(model_id, problem, max_new_tokens=384):
    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device

    prompt = make_prompt(tokenizer, problem)

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_inputs = {
        key: value.to(device)
        for key, value in prompt_inputs.items()
    }

    prompt_length = prompt_inputs["input_ids"].shape[1]

    # Generate a single reasoning trace.
    generated_ids = model.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )

    completion_ids = generated_ids[:, prompt_length:]

    if completion_ids.shape[1] == 0:
        raise ValueError("The model generated an empty completion.")

    # Re-run the full sequence to obtain token-aligned hidden states.
    # This avoids relying on the model-specific structure of
    # generation-time hidden_states.
    full_attention_mask = torch.ones_like(generated_ids)

    outputs = model(
        input_ids=generated_ids,
        attention_mask=full_attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # Each layer has shape:
    # (batch, sequence_length, hidden_dimension).
    #
    # Keep only generated-token positions, excluding the prompt.
    layer_states = [
        layer[0, prompt_length:, :]
        for layer in outputs.hidden_states[1:]
    ]

    # Shape: (layers, generated_tokens, hidden_dimension).
    hidden_states = torch.stack(layer_states, dim=0)

    result = hidden_states.float().cpu().numpy()

    del outputs, generated_ids, completion_ids
    del prompt_inputs, full_attention_mask, layer_states

    return result


def extract_features_and_series(model_id, problem):
    hidden_states = get_hidden_states(model_id, problem)

    # series: (channels, timesteps)
    series = trace_to_series(
        hidden_states,
        n_layers=8,
        length=128,
    )

    # features: (n_features,)
    features = trace_to_features(
        hidden_states,
        n_layers=8,
        length=128,
    )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(series, dtype=np.float32),
    )


def extract_features(model_id, problem):
    """Keep compatibility with callers expecting features only."""
    features, _ = extract_features_and_series(model_id, problem)
    return features
