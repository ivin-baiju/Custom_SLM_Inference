"""
Unit tests for sampling and generation strategies.
"""

import pytest
import torch
from runtime.request import GenerationConfig
from runtime.generation import sample_token


def test_greedy_sampling_deterministic():
    logits = torch.tensor([0.1, 0.5, 2.3, 0.2, 1.1])
    config = GenerationConfig(temperature=0.0)
    token = sample_token(logits, config)
    assert token == 2


def test_temperature_sampling():
    logits = torch.tensor([1.0, 2.0, 3.0])
    config = GenerationConfig(temperature=0.7, seed=42)
    token = sample_token(logits, config)
    assert token in [0, 1, 2]


def test_top_k_filtering():
    logits = torch.tensor([0.1, 0.2, 5.0, 0.05, 4.8])
    config = GenerationConfig(temperature=0.5, top_k=2, seed=123)
    token = sample_token(logits, config)
    # Must only be one of the top 2 indices (2 or 4)
    assert token in [2, 4]


def test_repetition_penalty():
    logits = torch.tensor([2.0, 2.0])
    # Token 0 was previously generated, penalize it
    config = GenerationConfig(temperature=0.0, repetition_penalty=2.0)
    token = sample_token(logits, config, generated_tokens=[0])
    # Token 0 is penalized, so token 1 should win
    assert token == 1
