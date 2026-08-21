# Chapter 1: Text Processing

Before a neural network can process linguistic data, the text must be converted into numerical representations via **tokenization**.

In this module, we implement a foundational tokenizer that:
1. Parses raw strings into discrete tokens (words and punctuation) using regular expressions.
2. Constructs a static vocabulary dictionary.
3. Assigns unique integer IDs to each token, handling Out-Of-Vocabulary (OOV) edge cases with a special `<|unk|>` token.
4. Encodes sequence data into integer tensors suitable for embedding lookups.

*Note: While we use a regex-based approach here for clarity, production LLMs typically employ sub-word tokenization algorithms like Byte-Pair Encoding (BPE) to manage infinite vocabularies with finite ID spaces and avoid sparsity.*

## Usage

Execute the script to observe the tokenization and decoding pipeline:
```bash
python tokenizer.py
```
