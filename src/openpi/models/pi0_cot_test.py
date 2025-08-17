import flax.nnx as nnx
import jax

import openpi.models.pi0_cot as _pi0_cot


def _get_frozen_state(config: _pi0_cot.Pi0CoTConfig) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_cot_embedder_frozen_no_lora():
    config = _pi0_cot.Pi0CoTConfig()
    state = _get_frozen_state(config)
    paths = list(state.keys())
    # Expect only the input embedding to be frozen when no LoRA is used
    for p in paths:
        print(p)
    assert len(paths) == 1
    assert all("input_embedding" in p for p in paths)


def test_pi0_cot_gemma_lora_freeze():
    config = _pi0_cot.Pi0CoTConfig(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    paths = list(state.keys())

    # LoRA adapters should never be frozen
    assert all("lora" not in p for p in paths)

    # Input embedding is always frozen
    assert any("input_embedding" in p for p in paths)

    # When freezing Gemma (paligemma), ensure we don't freeze action-expert ("_1") weights
    non_embedder = [p for p in paths if "input_embedding" not in p]
    assert len(non_embedder) > 0
    assert all("_1" not in p for p in non_embedder)


def test_pi0_cot_action_expert_lora_freeze():
    config = _pi0_cot.Pi0CoTConfig(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    paths = list(state.keys())

    # LoRA adapters should never be frozen
    assert all("lora" not in p for p in paths)

    # Input embedding is always frozen
    assert any("input_embedding" in p for p in paths)

    # When freezing the action expert, all other frozen params should be from the action expert ("_1")
    non_embedder = [p for p in paths if "input_embedding" not in p]
    assert len(non_embedder) > 0
    assert all("_1" in p for p in non_embedder)

if __name__ == "__main__":
    test_pi0_cot_embedder_frozen_no_lora()