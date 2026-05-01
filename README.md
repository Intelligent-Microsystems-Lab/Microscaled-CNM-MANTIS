# Microscaled-CNM-MANTIS
Official repository for "Foundry-SRAM-Compatible Compute-Near-Memory Macro for Microscaled Block-FP in 12-nm CMOS" (DAC 2026). Contains hardware simulation and evaluation code for MANTIS.

# MANTIS: Mixed-signal Near-memory Tensor Inference with MicroScaling

[![Conference](https://img.shields.io/badge/DAC-2026-blue.svg)](https://dac.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Foundry-SRAM-Compatible Compute-Near-Memory Macro for Microscaled Block-FP in 12-nm CMOS** > Samir Rahman, Arun M. George, Shehab Naga, Thomas Summe, Yipin Guo, Siddharth Joshi  
> *Design Automation Conference (DAC), 2026*

This repository contains the official hardware simulation and evaluation code for **MANTIS**, a foundry-SRAM-compatible, mixed-signal, Compute-Near-Memory (CNM) macro. MANTIS performs vector integer multiply-accumulate (MAC) operations in the analog charge domain while applying per-vector FP8 scaling factors digitally via microscaling (MXFP).

## 📖 Abstract
Trillion-parameter Transformers across vision, language, and action increasingly rely on reduced-precision floating-point (e.g., FP8) for dynamic range and efficiency. While their compute-intensive operations might make in-memory solutions attractive, the incompatibility of FP operations with analog computing renders many optimizations untenable for emerging compute-in/near-memory (CIM/CNM) platforms. 

To address this incompatibility, we present **MANTIS**. The proposed design, in a commercially available 12 nm node, reuses the digital-to-analog converter (DAC) in a successive approximation register (SAR) analog-to-digital converter (ADC) as both a bit-plane charge-domain accumulator and as part of the ADC. In end-to-end evaluations, our design maintains accuracy within ±0.3% of quantized baselines for billion-parameter LLMs, combining analog efficiency with FP-like dynamic range.

## 🗂️ Repository Structure

* `llama_mmlu_mxint_quantization_with_hw_simulation.ipynb`: PyTorch-based end-to-end evaluation pipeline. Simulates the MANTIS hardware dataflow (MXINT3 weights, MXINT8 activations, per-vector FP8 scaling) and injects characterized ADC noise into the Feed-Forward Networks (FFNs) of `Llama-3.1-8B-Instruct` to benchmark 5-shot MMLU accuracy.
* `adc_noise_molleding`: ADC distribution fitting code. 
* `benchmark_experiments`: Contains simulation code explained above. 

## 🚀 Getting Started

### Prerequisites
The simulation requires a GPU environment (A100 recommended for LLaMA-3.1-8B) and the following dependencies:
* `torch`
* `transformers`
* `datasets`
* `tqdm`
* `numpy`

*Note: You will need a valid Hugging Face token configured in your environment to download the LLaMA-3.1 model weights.*

### Running the Hardware Simulation
The provided notebook replaces standard PyTorch linear layers in the LLaMA MLP with a custom `SimulatedFFN` module. This module faithfully models the MANTIS compute flow:

1.  **Tiling & Quantization:** Inputs and weights are partitioned into $(32 \times 40)$ tiles and quantized to MXINT.
2.  **Analog MAC Simulation:** Integer mantissa products are accumulated.
3.  **Noise Injection:** Modeled ADC noise ($\mu=0.0, \sigma=0.001$, calibrated to LSB) is injected into the partial sums prior to digitization. 
4.  **Digital Recombination:** Per-vector FP8 scales are applied digitally to dequantize the noisy tile outputs.

To run the MMLU evaluation:
1. Open the Jupyter Notebook in your environment (e.g., Google Colab).
2. Ensure your `HF_TOKEN` is loaded in your environment secrets.
3. Execute the cells to patch the model FFNs and begin the 5-shot MMLU evaluation. Results will be logged to both standard output and your specified drive directory.

## 📊 Key Results
As detailed in the paper, MANTIS achieves:
* **7.62 ENOB** from the reused 8b SAR ADC.
* **Precision-Scalable Efficiency:** Ranging from 22.5 TOPS/W (MXINT3 $\times$ MXINT3) to 6.43 TOPS/W (MXINT8 $\times$ MXINT3).
* **LLM Resilience:** Minimal accuracy degradation ($\pm$ 0.3%) on zero-shot HellaSwag and 5-shot MMLU tasks when deploying microscaled activations and weights with hardware noise.

## 📝 Citation
If you find this code or our paper useful in your research, please cite our work:

```bibtex
@inproceedings{rahman2026mantis,
  title={Foundry-SRAM-Compatible Compute-Near-Memory Macro for Microscaled Block-FP in 12-nm CMOS},
  author={Rahman, Samir and George, Arun M. and Naga, Shehab and Summe, Thomas and Guo, Yipin and Joshi, Siddharth},
  booktitle={Proceedings of the 63rd ACM/IEEE Design Automation Conference (DAC)},
  year={2026}
}
