#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test chat with HF-formatted APT model (non-interactive)"""

import sys
import os
import re
import torch
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from apt.model.hf_compat.modeling_apt import APTForCausalLM


class SimpleCharTokenizer:
    """Simple character-level tokenizer for APT model"""
    def __init__(self):
        # Create base vocabulary
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
            '[EMOJI]': 4, '[PHRASE]': 5, '[EN]': 6, '[PY]': 7, '[JP]': 8, '[KR]': 9, '[MATH]': 10,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = 5000

        # Add character mappings
        self.char_to_id = self.vocab.copy()
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.next_id = 11  # Start from 11

        # Precompile regex for tags
        self.tag_pattern = re.compile(r'(\[EMOJI\]|\[PHRASE\]|\[EN\]|\[PY\]|\[JP\]|\[KR\]|\[MATH\])')

    def _tokenize_text(self, text):
        """Tokenize text by splitting tags first, then characters"""
        tokens = []
        # Split by tags
        parts = self.tag_pattern.split(text)
        for part in parts:
            if part in self.vocab:
                # If it's a tag, add ID directly
                tokens.append(self.vocab[part])
            else:
                # If it's regular text, process character by character
                for char in part:
                    if char.strip():
                        tokens.append(self._get_or_add_char(char))
                    elif char == ' ':  # Keep spaces
                        tokens.append(self._get_or_add_char(char))
        return tokens

    def _get_or_add_char(self, char):
        """Get character ID, add if not exists"""
        if char not in self.char_to_id:
            if self.next_id < self.vocab_size:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            else:
                return self.unk_token_id
        return self.char_to_id[char]

    def encode(self, text, return_tensors=None):
        """Encode text to ID sequence"""
        ids = [self.bos_token_id]
        ids.extend(self._tokenize_text(text))
        ids.append(self.eos_token_id)

        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode ID sequence to text"""
        chars = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                id = id.item()

            if skip_special_tokens and id in [self.pad_token_id, self.bos_token_id,
                                               self.eos_token_id, self.unk_token_id]:
                continue

            char = self.id_to_char.get(id, '[UNK]')
            chars.append(char)

        return ''.join(chars)


def load_tokenizer_from_checkpoint(checkpoint_path: str):
    """Load tokenizer from saved checkpoint"""
    print(f"Loading tokenizer from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    tokenizer = SimpleCharTokenizer()
    tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
    tokenizer.id_to_char = checkpoint['tokenizer_id_to_char']
    tokenizer.next_id = checkpoint['tokenizer_next_id']
    tokenizer.vocab_size = checkpoint['tokenizer_vocab_size']

    print(f"[OK] Tokenizer loaded successfully")
    print(f"  - Vocabulary size: {len(tokenizer.char_to_id)}")

    return tokenizer


def test_chat():
    """Test chat functionality with predefined inputs"""

    print("=" * 70)
    print("Testing HF APT Model Chat Functionality")
    print("=" * 70)

    model_path = "test_output/hf_model_test"
    checkpoint_path = "tests/saved_models/hlbd_model_20251222_034732.pt"

    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        model = APTForCausalLM.from_pretrained(model_path)
        model.eval()
        print(f"[OK] Model loaded successfully")
        print(f"  - Vocab size: {model.config.vocab_size}")
        print(f"  - Model type: {type(model).__name__}")
        device = next(model.parameters()).device
        print(f"  - Device: {device}")
    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load tokenizer
    print("\n" + "=" * 70)
    try:
        tokenizer = load_tokenizer_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"\n[ERROR] Tokenizer loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cases
    test_inputs = [
        "[EMOJI] ❤️",
        "[EN] I love you",
        "[PY] wǒ ài nǐ",
        "[JP] 愛してる",
        "[KR] 사랑해",
    ]

    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[Test {i}/{len(test_inputs)}]")
        print(f"Input: {user_input}")

        try:
            # Encode user input
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            input_ids = input_ids.to(device)

            # Generate response
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=model.config.pad_token_id,
                    eos_token_id=model.config.eos_token_id,
                )

            # Decode generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Remove input from output (show only generated part)
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):]

            print(f"Output: {generated_text}")
            print(f"[Generated {output.shape[1] - input_ids.shape[1]} new tokens]")

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_chat()
