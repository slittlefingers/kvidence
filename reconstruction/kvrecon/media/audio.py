"""Audio recovery entry point, invoked only after a verified audio-start token."""
from .solver import recover_latent


def build_problem(model, input_ids, media_positions, targets, token_types, **options):
    """Optimize audio encoder output before Gemma's audio adapter."""
    from .gemma4 import Gemma4Problem
    if model.config.audio_config is None or model.embed_audio is None:
        raise ValueError('Selected Gemma model has no audio adapter')
    config = model.config.audio_config
    # Encoder hidden size is 1024 in E2B; the adapter receives its 1536-wide
    # output projection, not the internal 1024-wide encoder state.
    width = config.output_proj_dims or config.hidden_size
    return Gemma4Problem(model, input_ids, media_positions, targets, token_types,
        modality='audio', adapter=model.embed_audio, adapter_input_dim=width, **options)


def recover(backend, request):
    if request['modality'] != 'audio':
        raise ValueError('Audio handler received a different modality')
    return recover_latent(backend, request, problem_factory=build_problem)


if __name__ == '__main__':
    from .pilot import main
    main('audio', build_problem)
