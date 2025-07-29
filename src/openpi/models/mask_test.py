import jax.numpy as jnp
import matplotlib.pyplot as plt


def visualize_mask(mask, title):
    plt.imshow(mask, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Key (attend to)")
    plt.ylabel("Query (generating)")
    plt.colorbar()
    plt.savefig(f"{title}.png")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def test_make_attn_mask():
    # Case 1: Pure causal attention
    input_mask = jnp.array([[1, 1, 1, 1, 1, 1]], dtype=bool)
    mask_ar = jnp.array([[1, 1, 1, 1, 1, 1]], dtype=bool)
    attn_mask = make_attn_mask(input_mask, mask_ar)[0]
    print("Pure causal attention")
    print("input_mask:", input_mask[0])
    print("mask_ar:   ", mask_ar[0])
    print("attn_mask:\n", attn_mask.astype(int))
    visualize_mask(attn_mask, "Pure causal attention")
    breakpoint()  # Set a breakpoint here to inspect the mask if needed

    # Case 2: Prefix-LM attention
    input_mask = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=bool)
    mask_ar = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=bool)
    attn_mask = make_attn_mask(input_mask, mask_ar)[0]
    print("Prefix-LM attention")
    print("input_mask:", input_mask[0])
    print("mask_ar:   ", mask_ar[0])
    print("attn_mask:\n", attn_mask.astype(int))
    visualize_mask(attn_mask, "Prefix-LM attention")
    breakpoint()  # Set a breakpoint here to inspect the mask if needed

    # Case 3: Blockwise attention
    input_mask = jnp.array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]], dtype=bool)
    mask_ar = jnp.array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]], dtype=bool)
    attn_mask = make_attn_mask(input_mask, mask_ar)[0]
    print("Blockwise attention")
    print("input_mask:", input_mask[0])
    print("mask_ar:   ", mask_ar[0])
    print("attn_mask:\n", attn_mask.astype(int))
    visualize_mask(attn_mask, "Blockwise attention")
    breakpoint()  # Set a breakpoint here to inspect the mask if needed


if __name__ == "__main__":
    test_make_attn_mask()
