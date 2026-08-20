import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Import architectural modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch01_text_processing.tokenizer import SimpleTokenizer
from ch03_simple_gpt.gpt_model import SimpleGPT

# Define a minimal training corpus to demonstrate overfitting dynamics
text_data = "hello world! this is a tiny dataset. we want to train our model to predict the next word."

# 1. Initialize and train the Tokenizer
# In a real environment, this is replaced by loading a pre-trained BPE (e.g. tiktoken)
tokenizer = SimpleTokenizer()
tokenizer.train(text_data)
token_ids = tokenizer.encode(text_data)
print(f"Vocabulary cardinality: {len(tokenizer.vocab)}")

# Prepare autoregressive context windows
# Input X and Target Y are shifted temporally by one position.
seq_len = 5
X, Y = [], []
for i in range(len(token_ids) - seq_len):
    X.append(token_ids[i:i+seq_len])
    Y.append(token_ids[i+1:i+seq_len+1])

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# 2. Instantiate the Decoder-only Transformer (GPT)
model = SimpleGPT(vocab_size=len(tokenizer.vocab), embed_dim=64, num_heads=4, num_layers=4, max_seq_len=seq_len)

# 3. Configure Optimization parameters
# AdamW is strictly preferred over standard Adam for Transformer architectures.
# AdamW decouples the weight decay from the gradient update, which is critical for regularization.
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# PyTorch's CrossEntropyLoss expects unnormalized logits as input, as it fuses LogSoftmax and NLLLoss 
# for superior numerical stability.
loss_fn = nn.CrossEntropyLoss()

# 4. Execute the Training Optimization Loop
epochs = 50
print("\nInitiating auto-regressive optimization loop...")
for epoch in range(epochs):
    # Enable gradient tracking
    model.train()
    
    # Forward pass: compute unnormalized log probabilities (logits)
    logits = model(X_tensor) # Shape: (batch_size, seq_len, vocab_size)
    
    # Restructure tensors to satisfy PyTorch's CrossEntropyLoss constraints (N, C)
    batch_size, sequence_length, vocab_size = logits.size()
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = Y_tensor.view(-1)
    
    # Compute cross-entropy loss (negative log likelihood of the true classes)
    loss = loss_fn(logits_flat, targets_flat)
    
    # Backpropagation via Automatic Differentiation
    optimizer.zero_grad(set_to_none=True) # set_to_none=True optimizes memory allocation vs zeroing tensors
    loss.backward()       
    
    # Note: Gradient clipping (torch.nn.utils.clip_grad_norm_) is highly recommended here for scale
    optimizer.step()      
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Cross-Entropy Loss: {loss.item():.4f}")

print("\nConvergence achieved. Transitioning to inference evaluation...")

# Autoregressive generation phase
context = "hello world!"
context_ids = tokenizer.encode(context)[-seq_len:]
context_tensor = torch.tensor([context_ids])

# Disable dropout and gradient tracking for deterministic inference
model.eval()
with torch.no_grad():
    # In a production environment, KV-Caching is strictly necessary here.
    # Without KV-Caching, attention recalculates for all past tokens, rendering inference O(N^2) instead of O(N).
    logits = model(context_tensor)
    
    # Extract logits strictly for the terminal token in the sequence
    last_word_logits = logits[0, -1, :]
    
    # Execute greedy decoding (argmax). Advanced samplers utilize Nucleus (Top-p) or Top-k sampling coupled with temperature.
    predicted_id = torch.argmax(last_word_logits).item()
    predicted_word = tokenizer.decode([predicted_id])

print(f"Context tensor state: '{context}'")
print(f"Model greedy prediction: '{predicted_word}'")
