# Modified from https://github.com/if-ai/ComfyUI-IF_Trellis/blob/main/IF_TrellisCheckpointLoader.py
import os
import logging
import torch
import huggingface_hub
import folder_paths
from trellis_model_manager import TrellisModelManager
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.backend_config import (
    set_attention_backend,
    set_sparse_backend,
    get_available_backends,
    get_available_sparse_backends
)
from typing import Literal
from torchvision import transforms

logger = logging.getLogger("IF_Trellis")

class IF_TrellisCheckpointLoader:
    """
    Node to manage the loading of the TRELLIS model with lazy backend selection.
    """
    def __init__(self):
        self.logger = logger
        self.model_manager = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # We might call these to figure out what's actually installed,
        # if we want to populate UI dropdowns:
        self.attn_backends = get_available_backends()         # e.g. { 'xformers': True, 'flash_attn': False, ... }
        self.sparse_backends = get_available_sparse_backends()# e.g. { 'spconv': True, 'torchsparse': True }

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types with device-specific options."""
        # Filter only available backends
        attn_backends = get_available_backends()
        sparse_backends = get_available_sparse_backends()

        # e.g. create a list of names that are True:
        available_attn = [k for k, v in attn_backends.items() if v]
        if not available_attn:
            available_attn = ['flash_attn']  # fallback

        available_sparse = [k for k, v in sparse_backends.items() if v]
        if not available_sparse:
            available_sparse = ['spconv']  # fallback

        return {
            "required": {
                "model_name": (["trellis-normal-v0-1"],),
                "dinov2_model": (["dinov2_vitl14_reg"],
                                 {"default": "dinov2_vitl14_reg",
                                  "tooltip": "Select which Dinov2 model to use."}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                #
                # The user picks from the actually installed backends
                #
                "attn_backend": (available_attn,
                                 {"default": "flash_attn" if "flash_attn" in available_attn else available_attn[0],
                                  "tooltip": "Select attention backend."}),
                "sparse_backend": (available_sparse,
                                   {"default": "spconv" if "spconv" in available_sparse else available_sparse[0],
                                    "tooltip": "Select sparse backend."}),
                "spconv_algo": (["implicit_gemm", "native", "auto"],
                                {"default": "implicit_gemm",
                                 "tooltip": "Spconv algorithm. 'implicit_gemm' is slower but more robust."}),
                "smooth_k": ("BOOLEAN",
                             {"default": True,
                              "tooltip": "Smooth-k for SageAttention. Only relevant if attn_backend=sage."}),
            },
        }

    RETURN_TYPES = ("TRELLIS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"

    def _setup_environment(self, attn_backend: str, sparse_backend: str, spconv_algo: str, smooth_k: bool):
        """
        Set up environment variables and backends lazily. 
        This is the main difference: we call our new lazy set_*_backend funcs.
        """
        # Try attention
        success = set_attention_backend(attn_backend)
        if not success:
            self.logger.warning(f"Failed to set {attn_backend} or not installed, fallback to sdpa.")

        # Try sparse
        success2 = set_sparse_backend(sparse_backend, spconv_algo)
        if not success2:
            self.logger.warning(f"Failed to set {sparse_backend} or not installed, fallback to default.")

        # If user wants SageAttn smooth_k, we set environment var (if they'd want that):
        os.environ['SAGEATTN_SMOOTH_K'] = '1' if smooth_k else '0'

    def _initialize_transforms(self):
        """Initialize image transforms if needed."""
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _optimize_pipeline(self, pipeline, use_fp16: bool = True):
        """
        Apply typical optimizations, half-precision, etc.
        """
        if self.device.type == "cuda":
            try:
                if hasattr(pipeline, 'cuda'):
                    pipeline.cuda()

                if use_fp16:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing(slice_size="auto")
                    if hasattr(pipeline, 'half'):
                        pipeline.half()
            except Exception as e:
                logger.warning(f"Some pipeline optimizations failed: {str(e)}")

        return pipeline

    def load_model(
        self,
        model_name: str,
        dinov2_model: str = "dinov2_vitl14_reg",
        attn_backend: str = "sdpa",
        sparse_backend: str = "spconv",
        spconv_algo: str = "implicit_gemm",
        use_fp16: bool = True,
        smooth_k: bool = True,
    ) -> tuple:
        """
        Load and configure the TRELLIS pipeline. 
        This is typically the main function invoked by ComfyUI at node execution time.
        """
        try:
            # 1) Setup environment + backends
            self._setup_environment(attn_backend, sparse_backend, spconv_algo, smooth_k)

            # 2) Get model paths, download if needed
            model_path = os.path.join(folder_paths.models_dir, "checkpoints", model_name)
            if not os.path.exists(model_path) or not os.listdir(model_path):
                repo_id = "Stable-X"
                try:
                    huggingface_hub.snapshot_download(
                        f"{repo_id}/{model_name}",
                        repo_type="model",
                        local_dir=model_path
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to download {repo_id}/{model_name} to: {model_path}, {e}")

            # 3) Create pipeline with the config
            pipeline = TrellisImageTo3DPipeline.from_pretrained(
                model_path,
                dinov2_model=dinov2_model
            )
            pipeline._device = self.device  # ensure pipeline uses our same device

            # 4) Apply optimizations
            pipeline = self._optimize_pipeline(pipeline, use_fp16)

            return (pipeline,)

        except Exception as e:
            logger.error(f"Error loading TRELLIS model: {str(e)}")
            raise
