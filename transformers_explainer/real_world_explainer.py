import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def run_explainer(input_text):
    print(f"Loading real GPT-2 model to process: '{input_text}'...")
    
    # 1. Load Model and Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Ensure model outputs attentions
    model.config.output_attentions = True
    model.eval()
    
    # 2. Tokenize Input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    tokens = [tokenizer.decode([i]).strip() for i in input_ids[0]]
    
    print(f"\n[Step 1] Tokenization:")
    for token, tid in zip(tokens, input_ids[0]):
        print(f"  '{token}' -> ID: {tid.item()}")
        
    # 3. Model Forward Pass
    print("\n[Step 2] Executing Forward Pass (Extracting Attention & Logits)...")
    with torch.no_grad():
        outputs = model(input_ids)
        
    # 4. Extract Attention Weights
    # outputs.attentions is a tuple of all layers. We take the last layer (Layer 12).
    # Shape of last_layer_attn: (batch=1, heads=12, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1]
    
    # Average the attention weights across all 12 heads to get a single heatmap representation
    avg_attention = last_layer_attn[0].mean(dim=0).numpy()
    
    # 5. Generate Heatmap Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_attention, annot=True, cmap="mako", xticklabels=tokens, yticklabels=tokens, fmt=".2f")
    plt.title("GPT-2 Last Layer Attention Weights (Averaged across heads)")
    plt.xlabel("Key Tokens (Attended To)")
    plt.ylabel("Query Tokens (Attending From)")
    
    heatmap_path = "attention_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, facecolor='#0d1117', edgecolor='none')
    print(f"\n[Step 3] Generated attention heatmap: {heatmap_path}")
    
    # 6. Extract Logits and Top-5 Predictions
    # We want the prediction for the *next* token, which is based on the logits of the *last* token in the sequence.
    next_token_logits = outputs.logits[0, -1, :]
    
    # Apply Softmax to convert raw logits into a probability distribution
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # Get top 5 probabilities and their indices
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    
    print("\n[Step 4] Top 5 Next Token Predictions:")
    predictions = []
    for i in range(top_k):
        pred_token = tokenizer.decode([top_indices[i].item()])
        pred_prob = top_probs[i].item() * 100
        predictions.append({'token': pred_token, 'prob': pred_prob})
        print(f"  {i+1}. '{pred_token}' ({pred_prob:.2f}%)")
        
    # 7. Generate HTML Report
    generate_html_report(input_text, tokens, heatmap_path, predictions)
    print("\n[Success] Generated real_explainer_report.html!")

def generate_html_report(input_text, tokens, heatmap_path, predictions):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-World GPT-2 Explainer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ background: #05070a; color: #e8f0e9; font-family: 'Inter', sans-serif; padding: 40px; line-height: 1.6; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1, h2 {{ color: #39ff96; font-family: 'JetBrains Mono', monospace; }}
        .card {{ background: #0d1117; border: 1px solid #21262d; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .tokens {{ display: flex; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 18px; margin-top: 10px; }}
        .token {{ background: #161b22; padding: 8px 12px; border: 1px solid #30363d; border-radius: 6px; }}
        img.heatmap {{ max-width: 100%; border-radius: 8px; border: 1px solid #30363d; }}
        .bar-container {{ display: flex; align-items: center; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; }}
        .bar-label {{ width: 120px; font-weight: bold; color: #4fd1ff; }}
        .bar-track {{ flex: 1; background: #161b22; height: 24px; border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: #39ff96; display: flex; align-items: center; padding-left: 8px; color: #000; font-weight: bold; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-World GPT-2 Explainer</h1>
        <p>This report was generated dynamically by hooking into a live PyTorch GPT-2 model.</p>
        
        <div class="card">
            <h2>1. Tokenization</h2>
            <p>Input Sequence: "{input_text}"</p>
            <div class="tokens">
"""
    for t in tokens:
        html_content += f'<div class="token">{t}</div>\n'
        
    html_content += f"""            </div>
        </div>

        <div class="card">
            <h2>2. Internal Attention Matrix (Final Layer)</h2>
            <p>This is the real $Q \\times K^T$ dot product matrix extracted from the model's final transformer block. It shows exactly how the model routed contextual information across the sequence.</p>
            <img class="heatmap" src="{heatmap_path}" alt="Attention Heatmap">
        </div>

        <div class="card">
            <h2>3. Next Token Logits & Probability</h2>
            <p>The model's final hidden state was projected against its 50,257-word vocabulary. A Softmax function converted these raw logits into the probability distribution shown below:</p>
            <div style="margin-top: 20px;">
"""
    for p in predictions:
        html_content += f"""
                <div class="bar-container">
                    <div class="bar-label">"{p['token']}"</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {p['prob']}%;">{p['prob']:.2f}%</div>
                    </div>
                </div>
"""
        
    html_content += """
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open("real_explainer_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    sentence = "The robot thinks"
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
    run_explainer(sentence)
