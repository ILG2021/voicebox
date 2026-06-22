"""
OmniVoice TTS backend implementation.

Wraps k2-fsa/OmniVoice for massively multilingual (600+ languages) zero-shot
voice cloning. Uses a diffusion language model-style architecture for high-quality
speech synthesis with fast inference.

Key design decisions:
- Voice prompt pattern: Deferred file paths (Pattern B) — OmniVoice re-encodes
  reference audio tokens internally on each generate() call using its
  HiggsAudioV2TokenizerModel. We avoid pre-computing tokens because the model
  must already be loaded for encoding, and caching the tokens would require
  storing large tensor dicts. Storing paths is simpler and memory-efficient.
- Device: CUDA → XPU → CPU. MPS is skipped because HiggsAudioV2TokenizerModel
  fails on MPS (output channels > 65536 limit).
- Sample rate: 24 kHz (set by the audio feature extractor).
- No needs_trim: OmniVoice includes postprocessing by default.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)

logger = logging.getLogger(__name__)

OMNIVOICE_HF_REPO = "k2-fsa/OmniVoice"

# ISO 639-1 → ISO 639-3 mapping for the languages Voicebox exposes.
# OmniVoice uses ISO 639-3 codes internally; the Voicebox API uses ISO 639-1.
_LANG_CODE_MAP = {
    "zh": "zh",   # Chinese (OmniVoice accepts "zh" directly)
    "en": "en",   # English
    "ja": "ja",   # Japanese
    "ko": "ko",   # Korean
    "de": "de",   # German
    "fr": "fr",   # French
    "ru": "ru",   # Russian
    "pt": "pt",   # Portuguese
    "es": "es",   # Spanish
    "it": "it",   # Italian
    "ar": "ar",   # Arabic
    "hi": "hi",   # Hindi
    "da": "da",   # Danish
    "nl": "nl",   # Dutch
    "pl": "pl",   # Polish
    "sv": "sv",   # Swedish
    "tr": "tr",   # Turkish
    "el": "el",   # Greek
    "fi": "fi",   # Finnish
    "he": "he",   # Hebrew
    "ms": "ms",   # Malay
    "no": "no",   # Norwegian
    "sw": "sw",   # Swahili
}


class OmniVoiceBackend:
    """OmniVoice multilingual TTS backend for zero-shot voice cloning."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device = None
        self._sample_rate: Optional[int] = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # MPS skipped: HiggsAudioV2TokenizerModel fails on MPS
        # (channels > 65536 unsupported). XPU is allowed for Intel Arc GPUs.
        return get_torch_device(allow_mps=False, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return OMNIVOICE_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(OMNIVOICE_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        """Load the OmniVoice model from HuggingFace."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading with progress tracking."""
        model_name = "omnivoice"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading OmniVoice on {device}...")

            import torch
            from omnivoice import OmniVoice

            # Map device string to device_map format expected by from_pretrained
            if device == "cuda":
                device_map = "cuda:0"
            elif device == "xpu":
                device_map = "xpu:0"
            else:
                device_map = "cpu"

            # Use float16 on GPU for efficiency; float32 on CPU (float16 on CPU
            # is not well-supported on all platforms)
            dtype = torch.float16 if device != "cpu" else torch.float32

            model = OmniVoice.from_pretrained(
                OMNIVOICE_HF_REPO,
                device_map=device_map,
                dtype=dtype,
            )
            model.eval()

            self.model = model
            self._sample_rate = getattr(model, "sampling_rate", 24000)

        logger.info(
            f"OmniVoice loaded successfully on {device} "
            f"(sample_rate={self._sample_rate})"
        )

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            self._sample_rate = None
            empty_device_cache(device)
            logger.info("OmniVoice unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        OmniVoice re-encodes reference audio internally at generation time
        using its HiggsAudioV2TokenizerModel, so we store the file path and
        reference text as a deferred prompt (Pattern B). No caching needed.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """Combine multiple reference audio samples by concatenation."""
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using OmniVoice.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict with optional 'ref_audio' and 'ref_text' keys.
            language: BCP-47 / ISO 639-1 language code.
            seed: Random seed (not directly supported by OmniVoice; sets torch seed).
            instruct: Voice design instructions (e.g. "female, british accent").

        Returns:
            Tuple of (audio_array, sample_rate) where audio_array is float32.
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text") or None

        # Validate reference audio path
        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None
            ref_text = None

        # Map ISO 639-1 → language name/code OmniVoice accepts
        omni_lang = _LANG_CODE_MAP.get(language, language)

        def _generate_sync():
            import torch

            if seed is not None:
                manual_seed(seed, self._device)

            logger.info(
                f"[OmniVoice] Generating: lang={language} "
                f"ref_audio={'yes' if ref_audio else 'no'} "
                f"instruct={'yes' if instruct else 'no'}"
            )

            with torch.inference_mode():
                # OmniVoice.generate() returns a list of np.ndarray (one per
                # text input). We always pass a single text so [0] is our result.
                kwargs = {
                    "text": text,
                    "language": omni_lang,
                }
                if ref_audio:
                    kwargs["ref_audio"] = ref_audio
                    if ref_text:
                        kwargs["ref_text"] = ref_text
                elif instruct:
                    # Voice design mode — no reference audio
                    kwargs["instruct"] = instruct

                audio_list = self.model.generate(**kwargs)

            # audio_list: list of np.ndarray with shape (T,) at sampling_rate
            audio = audio_list[0] if audio_list else np.zeros(1, dtype=np.float32)

            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            sample_rate = self._sample_rate or 24000
            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
