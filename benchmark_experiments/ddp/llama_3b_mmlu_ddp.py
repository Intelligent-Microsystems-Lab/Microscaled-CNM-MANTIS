import os
import argparse
import torch
import numpy as np
import random
import torch.nn as nn
import torch.distributed as dist
import datetime
import shutil
import json
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Quantization & Noise Simulation Functions ---

def quantize_per_group(tensor, mantissa_bits, group_size):
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    max_quant_val = (2**mantissa_bits) - 1
    original_shape = tensor.shape
    m_dim, k_dim = original_shape[-2], original_shape[-1]
    pad_size = 0
    if m_dim % group_size != 0:
        pad_size = group_size - (m_dim % group_size)
        tensor = nn.functional.pad(tensor, (0, 0, 0, pad_size))
        m_dim = tensor.shape[-2]

    grouped_tensor = tensor.view(*original_shape[:-2], m_dim // group_size, group_size, k_dim)
    max_abs_vals = torch.max(torch.abs(grouped_tensor), dim=-1, keepdim=True).values
    max_abs_vals = torch.max(max_abs_vals, dim=-2, keepdim=True).values
    sf = torch.where(max_abs_vals > 0, max_abs_vals / max_quant_val, 1.0)
    sf = sf.to(torch.bfloat16).to(tensor.dtype)
    sf.clamp_(min=1e-12)
    quantized_grouped_tensor = torch.round(grouped_tensor / sf)
    quantized_tensor = quantized_grouped_tensor.view(tensor.shape)

    if pad_size > 0:
        quantized_tensor = quantized_tensor[..., :original_shape[-2], :]

    sf_broadcastable = sf.repeat_interleave(group_size, dim=-3).view(*original_shape[:-1], 1)
    return quantized_tensor, sf_broadcastable

def add_adc_noise(tensor, mean, sigma, lsb):
    if sigma <= 0:
        return tensor
    noise_in_volts = torch.randn(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    noise_in_volts.mul_(sigma).add_(mean)
    integer_noise = torch.round_(noise_in_volts.div_(lsb))
    return tensor.add_(integer_noise)

# --- Simulated Hardware Layer ---

class SimulatedFFN(nn.Module):
    def __init__(self, original_mlp, tile_shape=(32, 40), noise_mean=0.0, noise_sigma=0.001, group_size=1):
        super().__init__()
        self.gate_proj_weight = nn.Parameter(original_mlp.gate_proj.weight)
        self.up_proj_weight = nn.Parameter(original_mlp.up_proj.weight)
        self.down_proj_weight = nn.Parameter(original_mlp.down_proj.weight)
        if original_mlp.gate_proj.bias is not None:
             self.gate_proj_bias = nn.Parameter(original_mlp.gate_proj.bias)
        else:
             self.gate_proj_bias = None

        v_min, v_max, levels = -0.5, 0.5, 256
        self.lsb = (v_max - v_min) / levels
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.tile_shape = tile_shape
        self.group_size = group_size

    def tiled_mat_mul_vec(self, input_tensor, weight_tensor):
        weight_tensor = weight_tensor.T
        m, k = input_tensor.shape
        _, n = weight_tensor.shape
        tile_m, block_size = self.tile_shape
        tile_n = tile_m

        pad_m = (tile_m - m % tile_m) % tile_m
        pad_k = (block_size - k % block_size) % block_size
        pad_n = (tile_n - n % tile_n) % tile_n

        inp_pad = nn.functional.pad(input_tensor, (0, pad_k, 0, pad_m))
        w_pad = nn.functional.pad(weight_tensor, (0, pad_n, 0, pad_k))
        m_pad, k_pad = inp_pad.shape
        _, n_pad = w_pad.shape

        inp_tiles = inp_pad.unfold(0, tile_m, tile_m).unfold(1, block_size, block_size)
        w_tiles = w_pad.unfold(0, block_size, block_size).unfold(1, tile_n, tile_n)
        inp_quant_tiles, inp_sf_tiles = quantize_per_group(inp_tiles, mantissa_bits=7, group_size=self.group_size)
        w_tiles_transposed = w_tiles.transpose(-2, -1)
        w_quant_tiles_transposed, w_sf_tiles = quantize_per_group(w_tiles_transposed, mantissa_bits=2, group_size=self.group_size)
        w_quant_tiles = w_quant_tiles_transposed.transpose(-2, -1)
        del inp_pad, w_pad, inp_tiles, w_tiles, w_tiles_transposed, w_quant_tiles_transposed

        accum_mantissas = torch.matmul(
            inp_quant_tiles.unsqueeze(2).to(torch.bfloat16),
            w_quant_tiles.unsqueeze(0).to(torch.bfloat16)
        )
        del inp_quant_tiles, w_quant_tiles
        noisy_accum_mantissas = add_adc_noise(accum_mantissas, self.noise_mean, self.noise_sigma, self.lsb)
        del accum_mantissas
        combined_sf = inp_sf_tiles.unsqueeze(2) * w_sf_tiles.transpose(-2, -1).unsqueeze(0)
        torch.mul(noisy_accum_mantissas, combined_sf, out=noisy_accum_mantissas)
        dequantized_tiles = noisy_accum_mantissas
        del combined_sf, inp_sf_tiles, w_sf_tiles, noisy_accum_mantissas
        output_tiles = torch.sum(dequantized_tiles, dim=1).to(input_tensor.dtype)
        del dequantized_tiles
        out_pad = output_tiles.permute(0, 2, 1, 3).contiguous().view(m_pad, n_pad)
        return out_pad[:m, :n]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_2d = x.view(-1, hidden_dim)
        gated_output = nn.functional.linear(x_2d, self.gate_proj_weight, self.gate_proj_bias)
        gated_states = nn.functional.silu(gated_output)
        up_states = self.tiled_mat_mul_vec(x_2d, self.up_proj_weight)
        activated_states = torch.mul(gated_states, up_states, out=gated_states)
        output_states_2d = self.tiled_mat_mul_vec(activated_states, self.down_proj_weight)
        return output_states_2d.view(batch_size, seq_len, -1)

def replace_ffn_with_simulation(model, noise_mean, noise_sigma, group_size):
    for layer in model.model.layers:
        original_mlp = layer.mlp
        layer.mlp = SimulatedFFN(original_mlp, noise_mean=noise_mean, noise_sigma=noise_sigma, group_size=group_size)
    return model

# --- Distributed Setup ---

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl")

def cleanup_distributed():
    dist.destroy_process_group()

# --- MMLU Evaluation Logic ---

def format_llama3_1_prompt(test_sample, subject_name, dev_samples=None, k_shot=0):
    system_prompt = f"You are an expert in {subject_name}. Please answer the following multiple choice question."
    messages = [{"role": "system", "content": system_prompt}]
    if k_shot > 0 and dev_samples:
        for i in range(k_shot):
            question = dev_samples[i]['question']
            choices_str = "\n".join([f"{chr(65+j)}. {dev_samples[i]['choices'][j]}" for j in range(len(dev_samples[i]['choices']))])
            user_content = f"Question: {question}\n{choices_str}\nAnswer:"
            messages.append({"role": "user", "content": user_content})
            assistant_content = f"{chr(65+dev_samples[i]['answer'])}"
            messages.append({"role": "assistant", "content": assistant_content})

    question = test_sample['question']
    choices_str = "\n".join([f"{chr(65+j)}. {test_sample['choices'][j]}" for j in range(len(test_sample['choices']))])
    final_user_content = f"Question: {question}\n{choices_str}\nAnswer:"
    messages.append({"role": "user", "content": final_user_content})
    correct_answer = chr(65 + test_sample['answer'])
    return messages, correct_answer

@torch.no_grad()
def evaluate_mmlu(model, tokenizer, device, checkpoint_dir, k_shot=5, batch_size=4):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    all_test_examples = []
    if rank == 0:
        print("Rank 0: Preparing and balancing the MMLU dataset...")
        mmlu_dataset = load_dataset("cais/mmlu", "all")
        dev_examples_by_subject = defaultdict(list)
        if k_shot > 0:
            for example in mmlu_dataset["dev"]:
                dev_examples_by_subject[example["subject"]].append(example)

        for subject_name in sorted(list(set(mmlu_dataset["test"]["subject"]))):
            subject_test_set = mmlu_dataset["test"].filter(lambda ex: ex["subject"] == subject_name, num_proc=4)
            subject_dev_samples = dev_examples_by_subject.get(subject_name)
            for test_sample in subject_test_set:
                messages, correct_answer = format_llama3_1_prompt(
                    test_sample, subject_name.replace("_", " "), dev_samples=subject_dev_samples, k_shot=k_shot
                )
                formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                all_test_examples.append({
                    "prompt": formatted_prompt,
                    "choices": test_sample['choices'],
                    "answer": correct_answer
                })
        random.shuffle(all_test_examples)
    
    object_list_to_broadcast = [all_test_examples]
    dist.broadcast_object_list(object_list_to_broadcast, src=0)
    all_test_examples = object_list_to_broadcast[0]

    examples_for_rank = all_test_examples[rank::world_size]
    
    local_correct = 0
    start_index = 0
    total_for_rank = len(examples_for_rank)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"rank_{rank}_checkpoint.json")
    if os.path.exists(checkpoint_path):
        print(f"Rank {rank}: Resuming from checkpoint {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data.get('progress', 0)
            local_correct = checkpoint_data.get('correct_count', 0)

    pbar = tqdm(range(start_index, total_for_rank, batch_size), 
                desc=f"GPU {rank} Evaluating", 
                position=rank,
                initial=start_index,
                total=total_for_rank)

    for i in pbar:
        batch_items = examples_for_rank[i:i+batch_size]
        if not batch_items:
            continue
        
        num_in_batch = len(batch_items)
        prompts = [item['prompt'] for item in batch_items]
        correct_answers = [item['answer'] for item in batch_items]
        
        # --- CRITICAL CHANGE FOR TORCH.COMPILE ---
        # Pad to a fixed max_length to ensure all input tensors have the same shape.
        # This prevents graph breaks and recompilations, maximizing performance.
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length", # Ensures stable tensor shapes
            truncation=True,
            max_length=2048
        ).to(device)
        
        outputs = model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]

        for j in range(num_in_batch):
            choices = batch_items[j]['choices']
            logits = last_token_logits[j]
            choice_tokens = [tokenizer.encode(f"{chr(65+k)}", add_special_tokens=False)[-1] for k in range(len(choices))]
            choice_logits = logits[choice_tokens]
            prediction = chr(65 + torch.argmax(choice_logits).item())
            if prediction == correct_answers[j]:
                local_correct += 1
        
        del inputs, outputs, last_token_logits, choice_logits
        
        if (i // batch_size) % 50 == 0 and i > start_index:
            with open(checkpoint_path, 'w') as f:
                json.dump({'progress': i + num_in_batch, 'correct_count': local_correct}, f)

    counts = torch.tensor([local_correct, total_for_rank], dtype=torch.long, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        total_correct = counts[0].item()
        total_all = counts[1].item()
        overall_accuracy = total_correct / total_all if total_all > 0 else 0
        print(f"\nOverall MMLU Accuracy ({k_shot}-shot): {overall_accuracy:.4f}")
        return {'overall_accuracy': overall_accuracy}
    return None

# --- Main Experiment Runner ---

def main():
    parser = argparse.ArgumentParser(description="Run distributed MMLU evaluation with hardware simulation.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face model ID.")
    parser.add_argument("--hf_token", type=str, default="hf_XXX", help="Your Hugging Face token.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation results.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of few-shot examples.")
    parser.add_argument("--noise_sigma", type=float, default=0.001, help="Std dev of ADC noise.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size PER GPU for MMLU evaluation.")
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ["HF_TOKEN"] = args.hf_token
    setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    group_sizes = [32, 16, 8, 1]
    all_experiment_results = {}

    for group_size in group_sizes:
        if rank == 0:
            print("\n" + "="*80)
            print(f"  STARTING EXPERIMENT: Group Size = {group_size}, Noise Sigma = {args.noise_sigma}  ")
            print("="*80 + "\n")

        checkpoint_dir_gs = os.path.join(args.checkpoint_dir, f"gs_{group_size}_noise_{args.noise_sigma}")
        if rank == 0:
            if os.path.exists(checkpoint_dir_gs):
                shutil.rmtree(checkpoint_dir_gs)
            os.makedirs(checkpoint_dir_gs, exist_ok=True)
        dist.barrier()

        if rank == 0: print(f"[{group_size=}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        if rank == 0: print(f"[{group_size=}] Loading base model...")
        # For torch.compile, use_cache=False is required.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, use_cache=False,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
        )

        if rank == 0: print(f"[{group_size=}] Replacing FFN layers...")
        quant_hw_model = replace_ffn_with_simulation(model, noise_mean=0.0, noise_sigma=args.noise_sigma, group_size=group_size)

        quant_hw_model.to(device)
        if rank == 0: print(f"[{group_size=}] Wrapping model with DDP...")
        model = DDP(quant_hw_model, device_ids=[local_rank])
        
        if rank == 0: print(f"[{group_size=}] Compiling the model (this may take a minute)...")
        model = torch.compile(model)
        if rank == 0: print(f"[{group_size=}] Model compiled successfully.")
        
        results = evaluate_mmlu(model, tokenizer, device, checkpoint_dir_gs, k_shot=args.k_shot, batch_size=args.batch_size)

        if rank == 0 and results:
            acc = results['overall_accuracy']
            all_experiment_results[group_size] = acc
            print(f"\n--- Results for Group Size {group_size}: Overall Accuracy = {acc:.4f} ---")

        del model, quant_hw_model, tokenizer
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print("\n" + "="*80)
        print(f"                OVERALL EXPERIMENT SUMMARY (Noise Sigma: {args.noise_sigma})")
        print("="*80 + "\n")
        for gs, accuracy in sorted(all_experiment_results.items(), key=lambda item: item[0], reverse=True):
            print(f"Group Size: {gs:<10} | Overall MMLU Accuracy: {accuracy:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"final_results_noise_{args.noise_sigma}.pt")
        print(f"\nSaving final summary to {output_file}")
        torch.save(all_experiment_results, output_file)

    cleanup_distributed()

if __name__ == "__main__":
    main()
