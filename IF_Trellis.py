# Modified from https://github.com/if-ai/ComfyUI-IF_Trellis/blob/main/IF_Trellis.py
import os
import torch
import imageio
import numpy as np
import logging
import traceback
from PIL import Image
import folder_paths
from typing import List, Union, Tuple, Literal, Optional, Dict
from easydict import EasyDict as edict
import gc
import comfy.model_management
import trimesh
import trimesh.exchange.export

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import MeshExtractResult

logger = logging.getLogger("IF_Trellis")

def get_subpath_after_dir(full_path: str, target_dir: str) -> str:
    try:
        full_path = os.path.normpath(full_path)
        full_path = full_path.replace('\\', '/')
        path_parts = full_path.split('/')
        try:
            index = path_parts.index(target_dir)
            subpath = '/'.join(path_parts[index + 1:])
            return subpath
        except ValueError:
            return path_parts[-1]
    except Exception as e:
        print(f"Error processing path in get_subpath_after_dir: {str(e)}")
        return os.path.basename(full_path)

class IF_TrellisImageTo3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRELLIS_MODEL",),
                "mode": (["single", "multi"], {"default": "single", "tooltip": "Mode. single is a single image. with multi you can provide multiple reference angles for the 3D model"}),
                "images": ("IMAGE", {"list": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 12.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01, "tooltip": "Simplify the mesh. the lower the value more polygons the mesh will have"}),
                "multimode": (["stochastic", "multidiffusion"], {"default": "stochastic"}),
                "project_name": ("STRING", {"default": "trellis_output"}),
            },
            "optional": {
                "masks": ("MASK", {"list": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("model_file", "video_path", "texture_image")
    FUNCTION = "image_to_3d"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    OUTPUT_NODE = True

    def __init__(self, vertices=None, faces=None, uvs=None, face_uvs=None, albedo=None):
        self.logger = logger
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.device = None
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.face_uvs = face_uvs
        self.albedo = albedo
        self.normals = None

    def torch_to_pil_batch(self, images: Union[torch.Tensor, List[torch.Tensor]],
                          masks: Optional[torch.Tensor] = None,
                          alpha_min: float = 0.1) -> List[Image.Image]:
        if isinstance(images, list):
            processed_tensors = []
            for img in images:
                if img.ndim == 3:
                    processed_tensors.append(img)
                elif img.ndim == 4:
                    processed_tensors.extend([t for t in img])
            images = torch.stack(processed_tensors, dim=0)

        logger.info(f"torch_to_pil_batch input shape: {images.shape}")
        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.shape[-1] != 3:
            if images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)

        processed_images = []
        for i in range(images.shape[0]):
            img = images[i].detach().cpu()
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    mask = masks[i] if i < masks.shape[0] else masks[0]
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    if mask.shape != img.shape[:2]:
                        import torch.nn.functional as F
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=img.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    if torch.any(mask > alpha_min):
                        mask = mask.to(dtype=img.dtype)
                        mask = mask.unsqueeze(-1) if mask.ndim == 2 else mask
                        img = torch.cat([img, mask], dim=-1)
                        mode = "RGBA"
                    else:
                        mode = "RGB"
                else:
                    mode = "RGB"
            else:
                mode = "RGB"
            img_np = (img.numpy() * 255).astype(np.uint8)
            processed_images.append(Image.fromarray(img_np, mode=mode))
            logger.info(f"Processed image {i}, shape: {img_np.shape}, mode: {mode}")

        return processed_images
    
    def generate_outputs(self, outputs, project_name,):
        out_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(out_dir, exist_ok=True)

        video_path = glb_path = ""
        texture_path = wireframe_path = ""
        texture_image = wireframe_image = None

        # Extract the first (and usually only) result
        mesh_output = outputs['mesh'][0]

        texture_path = None
        wireframe_path =  None
        glb_path = os.path.join(out_dir, f"{project_name}.glb")

        vertices = mesh_output.vertices.cpu().numpy()
        faces = mesh_output.faces.cpu().numpy()
        transformation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vertices = vertices @ transformation_matrix
        glb = trimesh.Trimesh(vertices, faces)
        glb.export(glb_path)
        glb_path = get_subpath_after_dir(glb_path, "output")
        full_glb_path = os.path.abspath(glb_path)
        logger.info(f"Full GLB path: {full_glb_path}, Processed GLB path: {glb_path}")

        # Create a blank texture if not saving or if texture mode is blank
        texture_image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Handle wireframe image
        if wireframe_path and os.path.exists(wireframe_path):
            wireframe_image = Image.open(wireframe_path).convert('RGB')
            wireframe_image = np.array(wireframe_image)
        else:
            wireframe_image = None

        # Clean up the large tensors after we're done using them
        del mesh_output
        torch.cuda.empty_cache()
        
        logger.info(f"Texture image shape: {texture_image.shape}")

        return video_path, glb_path, texture_path, wireframe_path, texture_image, wireframe_image

    def get_pipeline_params(self, seed, ss_sampling_steps, ss_guidance_strength,
                            slat_sampling_steps, slat_guidance_strength):
        if ss_sampling_steps < 1:
            raise ValueError("ss_sampling_steps must be >= 1")
        if slat_sampling_steps < 1:
            raise ValueError("slat_sampling_steps must be >= 1")
        if ss_guidance_strength < 0:
            raise ValueError("ss_guidance_strength must be >= 0")
        if slat_guidance_strength < 0:
            raise ValueError("slat_guidance_strength must be >= 0")

        return {
            "seed": seed,
            "formats": ["mesh"],
            "preprocess_image": True,
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            "slat_sampler_params": {
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            }
        }

    @torch.inference_mode()
    def image_to_3d(
        self,
        model: TrellisImageTo3DPipeline,
        mode: str,
        images: torch.Tensor,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        mesh_simplify: float,
        multimode: str,
        project_name: str,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str, torch.Tensor]:
        try:
            logger.info(f"Input images tensor initial shape: {images.shape}")
            with model.inference_context():
                self.mesh_simplify = mesh_simplify
                self.device = model.device

                pipeline_params = self.get_pipeline_params(
                    seed, ss_sampling_steps, ss_guidance_strength,
                    slat_sampling_steps, slat_guidance_strength,
                )

                # Handle single vs multi mode differently
                if mode == "single":
                    # Take just the first image regardless of how many were input
                    images = images[0:1]
                    pil_imgs = self.torch_to_pil_batch(images, masks)
                    outputs = model.run(pil_imgs[0],
                    **pipeline_params)
                else:
                    # In multi mode, treat the whole list as a batch
                    pil_imgs = self.torch_to_pil_batch(images, masks)
                    logger.info(f"Processing {len(pil_imgs)} views for multi-view reconstruction")
                    outputs = model.run_multi_image(
                        pil_imgs,
                        mode=multimode,
                        **pipeline_params
                    )

                video_path, glb_path, _, _, texture_image, _ = self.generate_outputs(
                    outputs,
                    project_name,
                )

                # Fallback to black texture
                texture_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

                self.cleanup_outputs(outputs)
                return glb_path, video_path, texture_tensor

        except Exception as e:
            logger.error(f"Error in image_to_3d: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def cleanup_outputs(self, outputs):
        # Now we only need to clean up the dictionary itself
        del outputs
        gc.collect()
