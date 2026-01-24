#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) - Text Generator

This module contains functions for generating natural text using APT models.
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
F = torch.nn.functional
import logging

def generate_natural_text(model, tokenizer, prompt, max_steps=50, temperature=0.7, top_p=0.9):
    """
    Generate natural text using an APT model
    
    Args:
        model: The APT model for text generation
        tokenizer: Tokenizer for encoding/decoding text
        prompt (str): Input prompt for text generation
        max_steps (int): Maximum number of generation steps
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter (higher = more diverse)
        
    Returns:
        tuple: (generated_text, output_ids, temperature, top_p)
            - generated_text (str): The generated text
            - output_ids (torch.Tensor): Token IDs of the generated text
            - temperature (float): The temperature used for generation
            - top_p (float): The top_p value used for generation
    """
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.size(1) + max_steps,
        temperature=temperature,
        top_p=top_p
    )
    
    # Decode the generated text
    try:
        # 安全的解码，处理中文字符
        generated_text = safe_decode(tokenizer, output_ids[0])
    except Exception as e:
        logger = logging.getLogger('apt_model')
        logger.warning(f"标准解码失败: {e}，尝试备用解码方法")
        
        # 备用解码方法
        generated_text = custom_decode(tokenizer, output_ids[0])
    
    return generated_text, output_ids, temperature, top_p

def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """
    安全的解码函数，处理中文和其他特殊字符
    
    Args:
        tokenizer: 分词器
        token_ids: 要解码的token ID列表
        skip_special_tokens: 是否跳过特殊标记
    
    Returns:
        str: 解码后的文本
    """
    try:
        # 首先尝试标准解码
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except KeyError as e:
        # 如果是ChineseTokenizer，使用自定义解码
        if hasattr(tokenizer, 'decoder'):
            return custom_decode(tokenizer, token_ids, skip_special_tokens)
        # 否则尝试逐个解码
        text = ""
        for token_id in token_ids.tolist():
            try:
                if skip_special_tokens and token_id in tokenizer.all_special_ids:
                    continue
                token = tokenizer.convert_ids_to_tokens(token_id)
                text += token
            except:
                # 如果解码失败，添加占位符
                text += "[?]"
        return text

def custom_decode(tokenizer, token_ids, skip_special_tokens=True):
    """
    为ChineseTokenizer自定义的解码函数
    
    Args:
        tokenizer: 分词器
        token_ids: 要解码的token ID列表
        skip_special_tokens: 是否跳过特殊标记
    
    Returns:
        str: 解码后的文本
    """
    if hasattr(tokenizer, 'decoder'):
        # 对于自定义的ChineseTokenizer
        text = ""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        for token_id in token_ids:
            if skip_special_tokens and hasattr(tokenizer, 'special_tokens_map') and \
               token_id in tokenizer.special_tokens_map.values():
                continue
                
            if token_id in tokenizer.decoder:
                text += tokenizer.decoder[token_id]
            else:
                # 未知ID，添加占位符
                text += "[UNK]"
        return text
    else:
        # 对于标准的tokenizer，使用一个简单的备用方法
        return " ".join([str(id) for id in token_ids.tolist()])

def generate_with_context(model, tokenizer, context, max_steps=50, temperature=0.7, top_p=0.9):
    """
    Generate text with conversation context
    
    Args:
        model: The APT model for text generation
        tokenizer: Tokenizer for encoding/decoding text
        context (list): List of conversation turns
        max_steps (int): Maximum number of generation steps
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter (higher = more diverse)
        
    Returns:
        str: The generated response text
    """
    # Prepare the context as input
    if len(context) > 1:
        # Use a limited window of context to avoid exceeding model's context length
        recent_context = context[-min(6, len(context)):]
        prompt = "\n".join(recent_context)
    else:
        prompt = context[0] if context else ""
    
    # Generate the text
    response, _, _, _ = generate_natural_text(
        model, tokenizer, prompt, 
        max_steps=max_steps, 
        temperature=temperature,
        top_p=top_p
    )
    
    # Extract the model response part (removing input context)
    if "User: " in response:
        # Remove the input part, keeping only the response
        response = response.split("User: ")[-1]
    
    # Clean up the response text
    response = response.strip()
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def beam_search_generation(model, tokenizer, prompt, num_beams=5, max_steps=50):
    """
    Generate text using beam search for higher quality
    
    Args:
        model: The APT model for text generation
        tokenizer: Tokenizer for encoding/decoding text
        prompt (str): Input prompt for text generation
        num_beams (int): Number of beams for beam search
        max_steps (int): Maximum number of generation steps
        
    Returns:
        str: The generated text
    """
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text using beam search
    with torch.no_grad():
        # Implement beam search generation logic
        # Note: This is a simplified implementation that should be expanded
        # in an actual implementation
        
        # Start with the input ids for all beams
        beam_scores = torch.zeros(1, num_beams, device=device)
        beam_sequences = input_ids.repeat(num_beams, 1)
        
        for _ in range(max_steps):
            # Forward pass
            outputs = model(beam_sequences, beam_sequences)
            next_token_logits = outputs[:, -1, :]
            
            # Calculate log probabilities
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Add score to log probabilities
            next_scores = beam_scores.view(-1, 1) + log_probs
            
            # Get the top k scores and indices
            topk_scores, topk_indices = next_scores.view(-1).topk(num_beams, sorted=True)
            
            # Convert indices to token ids and beam indices
            beam_indices = topk_indices // next_scores.shape[1]
            token_indices = topk_indices % next_scores.shape[1]
            
            # Update sequences
            next_sequences = []
            for beam_idx, token_idx in zip(beam_indices, token_indices):
                next_sequences.append(
                    torch.cat([beam_sequences[beam_idx], token_idx.unsqueeze(0)], dim=0)
                )
            
            beam_sequences = torch.stack(next_sequences)
            beam_scores = topk_scores.unsqueeze(0)
        
        # Return the highest scoring beam
        best_sequence = beam_sequences[0]
    
    # Decode the generated text with safe decoder
    generated_text = safe_decode(tokenizer, best_sequence)
    
    return generated_text

def batch_generate(model, tokenizer, prompts, max_steps=50, temperature=0.7, top_p=0.9):
    """
    Generate text for multiple prompts in batch
    
    Args:
        model: The APT model for text generation
        tokenizer: Tokenizer for encoding/decoding text
        prompts (list): List of input prompts
        max_steps (int): Maximum number of generation steps
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter
        
    Returns:
        list: List of generated texts
    """
    device = next(model.parameters()).device
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = 4  # Adjust based on available GPU memory
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Encode all prompts
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Generate text for all inputs
        with torch.no_grad():
            batch_outputs = model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_length=batch_inputs.input_ids.size(1) + max_steps,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode all outputs safely
        batch_results = []
        for sequence in batch_outputs.sequences:
            batch_results.append(safe_decode(tokenizer, sequence))
        results.extend(batch_results)
    
    return results