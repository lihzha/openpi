import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


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


def cross_entropy_loss(
    logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray | None = None, axis: int = -1, train: bool = True
) -> jnp.ndarray:
    """
    Args
    ----
      logits : (..., V)   – raw scores.
      labels : (...)      – int32 / int64 class‑ids, same leading shape as logits without the class dim.
      mask   : (...) or None – 0/1 or bool; broadcastable to `labels`.
      axis   : int        – class dimension in `logits`.
      train  : bool       – if True → mean loss, else → summed loss.

    Returns
    -------
      scalar mean (train=True) or scalar sum (train=False).
    """
    # log‑probs
    log_probs = nnx.log_softmax(logits, axis=axis)  # (..., V)

    # gather log‑prob of the gold class
    gather_idx = jnp.expand_dims(labels.astype(jnp.int32), axis=axis)  # (..., 1)
    gold_logp = jnp.take_along_axis(log_probs, gather_idx, axis=axis)  # (..., 1)
    loss = -gold_logp.squeeze(axis)  # (...)

    # optional masking
    if mask is not None:
        loss = loss * mask
        denom = jnp.maximum(mask.sum(), 1)  # avoid ÷0 for empty mask
    else:
        denom = loss.size

    total = loss.sum()
    return total / denom if train else total


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0CoTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = (
        300  # TODO: Maximum length of the tokenized prompt, including reasoning tokens. Was 48 for prompt-only.
    )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0CoT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0CoT":
        return Pi0CoT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_reasoning_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0CoT(_model.BaseModel):
    EOS_ID = 1  # TODO: hard-coded for PaliGemma
    lang_action_only: bool = True

    def __init__(self, config: Pi0CoTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # self.lang_action_only = config.lang_action_only

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        tokens = []
        _img_ar_masks = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other. broadcast to (B, S)
            _img_ar_masks += [False] * image_tokens.shape[1]
        img_ar_mask = jnp.array(_img_ar_masks)
        img_ar_mask = einops.repeat(img_ar_mask, "s -> b s", b=image_tokens.shape[0])

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            text_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(text_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs. reasoning tokens casual attention.
            text_ar_mask = obs.tokenized_reasoning_mask

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate([img_ar_mask, text_ar_mask], axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # TODO: assume reasoning is already tokenized for compute_loss. Need to tokenize reasoning on-the-fly for inference.
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        if not self.lang_action_only:
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
            suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])
            # one big forward pass of prefix + suffix at once. Reasoning is identical to prompt except for ar_mask.
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        else:
            suffix_tokens = None
            input_mask = prefix_mask
            ar_mask = prefix_ar_mask

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        # TODO: should we jointly compute loss on the prompt?
        shift_labels = observation.tokenized_prompt[:, 1:]  # shape (B, max_len-1). TODO: contiguous?
        max_len = observation.tokenized_reasoning_mask.shape[1]
        shift_tokens = prefix_out[:, -max_len:-1, :]  # shape (B, max_len-1, D)
        shift_logits = self.PaliGemma.llm(shift_tokens, method="decode")  # shape (B, max_len-1, V)
        # compute cross-entropy loss on the reasoning tokens
        reasoning_and_pad_mask = jnp.logical_and(
            observation.tokenized_reasoning_mask[:, 1:], observation.tokenized_prompt_mask[:, 1:]
        )
        loss = cross_entropy_loss(shift_logits, shift_labels, mask=reasoning_and_pad_mask, axis=-1, train=True)

        if not self.lang_action_only:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            loss += jnp.mean(jnp.square(v_t - u_t))

        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, kv_cache = self._sample_reasoning_tokens(observation)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # TODO: assert bs=1, i.e. assuming only used for inference.
    def _sample_reasoning_tokens(self, observation: _model.Observation) -> _model.Actions:
        # ───────────────── 1. Prefix pass ─────────────────
        p_tokens, p_mask0, p_ar_mask0 = self.embed_prefix(observation)  # (B,Tp,D) + (B,Tp)
        b, tp, d = *p_tokens.shape[:2], p_tokens.shape[-1]
        gen_len = observation.tokenized_prompt.shape[1]  # generation length
        max_len = gen_len + tp  # generation budget

        # Full-length (static-shape) buffers
        p_mask = jnp.zeros((b, max_len), dtype=bool).at[:, :tp].set(p_mask0)
        p_ar_mask = jnp.zeros((b, max_len), dtype=bool).at[:, :tp].set(p_ar_mask0)

        # prefix attention & positions
        pref_attn = make_attn_mask(p_mask[:, :tp], p_ar_mask[:, :tp])  # (B,Tp,Tp)
        pos_pref = jnp.cumsum(p_mask[:, :tp], axis=1) - 1

        (hs, _), kv0 = self.PaliGemma.llm([p_tokens, None], mask=pref_attn, positions=pos_pref)
        curr_h = hs[:, -1:, :]  # (B,1,D)
        curr_id = jnp.argmax(self.PaliGemma.llm(curr_h, method="decode"), axis=-1)  # (B,1)

        # ───────────────── 2. Static KV cache ─────────────
        nl, _, _, k, h = kv0[0].shape  # num_kv_heads, head_dim
        k_cache = jnp.zeros((nl, b, max_len, k, h), dtype=kv0[0].dtype).at[:, :, :tp].set(kv0[0])
        v_cache = jnp.zeros_like(k_cache).at[:, :, :tp].set(kv0[1])

        # ───────────────── 3. Output buffers ──────────────
        h_buf = jnp.zeros((b, gen_len, d), dtype=hs.dtype).at[:, 0].set(curr_h.squeeze(1))
        id_buf = jnp.zeros((b, gen_len, 1), dtype=jnp.int32).at[:, 0].set(curr_id)

        t0 = 0

        # ───────────────── 4. Body / Cond ────────────────
        def step(carry):
            (curr_h, curr_id, k_cache, v_cache, p_mask, p_ar_mask, h_buf, id_buf, _t) = carry

            t_abs = _t + tp  # tp is the (static) prefix length
            jax.debug.print("t:{t}", t=_t)

            p_mask = p_mask.at[:, t_abs].set(True)
            p_ar_mask = p_ar_mask.at[:, t_abs].set(True)

            attn_row = make_attn_mask(p_mask, p_ar_mask)[:, -1:, :]  # (B,1,MAX)
            attn_row = jax.lax.dynamic_update_slice(
                jnp.zeros((p_mask.shape[0], 1, p_mask.shape[1] + 1), dtype=bool),  # STATIC big tensor
                attn_row,  # STATIC window
                (0, 0, 0),  # start indices can vary
            )
            pos = jnp.full((p_mask.shape[0], 1), t_abs, dtype=jnp.int32)

            (next_h, _), kv_new = self.PaliGemma.llm(
                [curr_h, None],
                positions=pos,  # (B,1)
                mask=attn_row,  # (B,1,MAX)
                kv_cache=(k_cache, v_cache),
            )

            next_id = jnp.argmax(self.PaliGemma.llm(next_h, method="decode"), axis=-1)
            jax.debug.print("next_id:{next_id}", next_id=next_id)

            k_cache = k_cache.at[:, :, t_abs].set(kv_new[0][:, :, -1])
            v_cache = v_cache.at[:, :, t_abs].set(kv_new[1][:, :, -1])

            _t += 1
            h_buf = h_buf.at[:, _t].set(next_h.squeeze(1))
            id_buf = id_buf.at[:, _t].set(next_id)

            return (next_h, next_id, k_cache, v_cache, p_mask, p_ar_mask, h_buf, id_buf, _t)

        def cond(carry):
            _, curr_id, _, _, _, _, _, _, t = carry
            unfinished = jnp.any(curr_id != self.EOS_ID)
            return jnp.logical_and(unfinished, t < gen_len - 1)

        # ───────────────── 5. While-loop ─────────────────
        carry = (curr_h, curr_id, k_cache, v_cache, p_mask, p_ar_mask, h_buf, id_buf, t0)
        curr_h, curr_id, k_cache, v_cache, p_mask, p_ar_mask, h_buf, id_buf, t = jax.lax.while_loop(cond, step, carry)

        # ───────────────── 6. Pack outputs ───────────────
        # final_len = t
        # final_mask = make_attn_mask(p_mask[:, : final_len + tp], p_ar_mask[:, : final_len + tp])
        # final_tokens = jnp.concatenate([p_tokens, h_buf[:, :final_len]], axis=1)
        # out_ids = id_buf[:, :final_len, 0]

        return p_mask, p_ar_mask, h_buf, id_buf, t, k_cache, v_cache

    @override
    def sample_reasoning(self, observation: _model.Observation):
        p_mask, p_ar_mask, h_buf, logits, t, k_cache, v_cache = self._sample_reasoning_tokens(observation)
        # return self.PaliGemma.llm(reasoning_tokens, method="decode")  # logits
        return logits, t, k_cache, p_mask, p_ar_mask
