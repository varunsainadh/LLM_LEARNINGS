# Large Language Models & Transformer Architectures (From Scratch)

Welcome to the repository for understanding and building Large Language Models (LLMs) and core Transformer architectures from scratch.

This repository is designed for developers seeking an intermediate-level deep dive into the underlying mechanics of modern AI. We will walk through the core components step-by-step, starting from text tokenization, traversing the intricacies of the self-attention mechanism, and culminating in a functional Decoder-only Transformer (GPT) model.

## 📚 Chapters & Architecture Walkthrough

- **[Chapter 1: Text Processing](ch01_text_processing/)** - Implementation of tokenization, vocabulary construction, and mapping sequence spaces to integer IDs.
- **[Chapter 2: Attention Mechanism](ch02_attention_mechanism/)** - An in-depth look at Scaled Dot-Product Attention ($QK^T/\sqrt{d_k}$) and the Multi-Head Attention formulations that allow Transformers to capture long-range dependencies.
- **[Chapter 3: The Transformer Architecture](ch03_simple_gpt/)** - Assembling the token embeddings, positional encodings, and stacked attention blocks into a GPT (Decoder-only Transformer) model.
- **[Chapter 4: Training Loop & Optimization](ch04_training_loop/)** - Formulating the auto-regressive next-token prediction task using Cross-Entropy loss and the AdamW optimizer.

## 🚀 Getting Started

1. Clone this repository (or download it).
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to `ch01_text_processing` and review the source code to begin.

## 📖 Detailed Explanation

For a comprehensive, detailed walkthrough of this codebase and the theory behind Transformers, please open [docs/llm_explanation_report.html](docs/llm_explanation_report.html) in your web browser.
