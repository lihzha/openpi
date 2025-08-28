import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_cot_tokenize():
    tok = _tokenizer.PaligemmaTokenizer(max_len=200, include_decimal_point=True, left_pad=True)
    prompt = "Pick up the red block\n"
    reasoning = "Move left 3.10cm and move up 3.12cm and move down 129.3cm and set gripper to 0.1"
    tokens, attn_mask, reasoning_mask, numeric_mask = tok.tokenize_cot(prompt, reasoning)
    assert tokens.shape == (200,)
    assert attn_mask.shape == (200,)
    assert reasoning_mask.shape == (200,)
    assert numeric_mask.shape == (200,)
    breakpoint()

if __name__ == "__main__":
    test_cot_tokenize()