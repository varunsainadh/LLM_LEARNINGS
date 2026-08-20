import re

class SimpleTokenizer:
    """
    A foundational Tokenizer implementation.
    Maps string sequences to discrete integer ID spaces.
    """
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        
    def train(self, text):
        """
        Constructs the vocabulary from a training corpus.
        """
        # Regex split on punctuation and whitespace boundaries
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        
        # Deduplicate and sort to form the vocabulary base
        unique_tokens = sorted(list(set(tokens)))
        
        # Inject special token to handle Out-Of-Vocabulary (OOV) instances
        unique_tokens.append("<|unk|>")
        
        for i, token in enumerate(unique_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
            
    def encode(self, text):
        """
        Encodes a raw string into a list of token IDs.
        """
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        
        token_ids = []
        for token in tokens:
            # Handle OOV by assigning the unknown token ID
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<|unk|>"])
        return token_ids
        
    def decode(self, token_ids):
        """
        Decodes a sequence of token IDs back into a human-readable string.
        """
        text = " ".join([self.inverse_vocab[i] for i in token_ids])
        # Regex cleanup for leading spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    sample_text = "Hello, world! This is a foundational tokenizer for our LLM implementation."
    print("Original text:", sample_text)
    
    tokenizer = SimpleTokenizer()
    tokenizer.train(sample_text)
    print("\nVocabulary size:", len(tokenizer.vocab))
    
    ids = tokenizer.encode("Hello, world!")
    print("\nEncoded 'Hello, world!':", ids)
    
    decoded = tokenizer.decode(ids)
    print("Decoded sequence:", decoded)
