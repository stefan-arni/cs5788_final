import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_utils import get_dtype, pil_to_tensor, tensor_to_pil, encode_image, decode_latent, create_composite
from utils.mask_utils import prepare_latent_mask, gaussian_blur_mask
from inversion.ddpm_inversion import ddpm_invert, reconstruct_xt
from noise_shift.noise_shift import shift_all_noise_maps


MODEL_ID = "sd2-community/stable-diffusion-2-1-base"
INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"


class ObjectRelocationPipeline:
    """Texture-preserving object relocation via DDPM noise prior shift.

    Pipeline: pixel-space composite -> DDPM inversion -> noise-map shift
    source->target -> SDEdit with background + source cleanup locks.
    Ablation: use_noise_shift=False drops the shift step.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        inpaint_model_id: str = INPAINT_MODEL_ID,
        device: torch.device = None,
        local_files_only: bool = False,
    ):
        if device is None:
            from utils.image_utils import get_device
            device = get_device()
        self.device = device
        self.model_id = model_id
        self.inpaint_model_id = inpaint_model_id
        self.local_files_only = local_files_only
        dtype = get_dtype(device)
        self.dtype = dtype

        print(f"Loading SD 2.1 on {device} ({dtype})...")
        kw = dict(local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kw).to(device=device, dtype=dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kw).to(device=device, dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kw).to(device=device, dtype=dtype)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **kw)
        self.prediction_type = self.scheduler.config.prediction_type
        # Cap at 512 on MPS to avoid OOM; Colab (CUDA) uses the native 768
        native_size = self.unet.config.sample_size * 8
        self.image_size = 512 if device.type == "mps" else native_size
        self.latent_size = self.image_size // 8
        self.inpaint_pipe = None
        print(f"All components loaded. Prediction type: {self.prediction_type}, image size: {self.image_size}")

    def _make_generator(self, seed: int) -> torch.Generator:
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=gen_device).manual_seed(seed)

    def _load_inpaint_pipe(self):
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe

        print(f"Loading inpainting model on {self.device} ({self.dtype})...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.inpaint_pipe.set_progress_bar_config(disable=True)
        if hasattr(self.inpaint_pipe, "enable_attention_slicing"):
            self.inpaint_pipe.enable_attention_slicing()
        return self.inpaint_pipe

    def _masks_are_effectively_same(
        self,
        source_mask: Image.Image,
        target_mask: Image.Image,
    ) -> bool:
        src = np.array(source_mask.convert("L")) > 127
        tgt = np.array(target_mask.convert("L")) > 127
        union = np.logical_or(src, tgt).sum()
        if union == 0:
            return True
        iou = np.logical_and(src, tgt).sum() / union
        return iou > 0.98

    def _shift_pixels_to_target(
        self,
        original: Image.Image,
        source_mask: Image.Image,
        target_mask: Image.Image,
    ) -> Image.Image:
        """Copy original object pixels to the target centroid position.
        Returns a copy of `original` with the object also pasted at the target
        location (same centroid translation used by create_composite).
        """
        orig_arr = np.array(original.convert("RGB"))
        src = np.array(source_mask.convert("L")) > 127
        tgt = np.array(target_mask.convert("L")) > 127
        sy, sx = np.where(src)
        ty, tx = np.where(tgt)
        if len(sy) == 0 or len(ty) == 0:
            return original.copy()
        dy = int(round(ty.mean() - sy.mean()))
        dx = int(round(tx.mean() - sx.mean()))
        H, W = orig_arr.shape[:2]
        texture = orig_arr.copy()
        valid = (sy + dy >= 0) & (sy + dy < H) & (sx + dx >= 0) & (sx + dx < W)
        texture[sy[valid] + dy, sx[valid] + dx] = orig_arr[sy[valid], sx[valid]]
        return Image.fromarray(texture)

    def _compose_final(
        self,
        original: Image.Image,
        source_fill: Image.Image,
        first_pass: Image.Image,
        texture_layer: Image.Image,
        source_mask: Image.Image,
        target_mask: Image.Image,
        feather_sigma: float,
        texture_alpha: float = 0.75,
    ) -> Image.Image:
        """Four-way blend:
        - target region → texture_alpha * texture_layer + (1-texture_alpha) * first_pass
          (original sharp pixels blended with SDEdit harmonization)
        - source region → source_fill (SD inpainting output with neutral prompt — no animal)
        - background    → original   (untouched pixels preserved exactly)
        Weights sum to 1 everywhere; target takes priority over source in any overlap.
        """
        orig_arr = np.array(original.convert("RGB")).astype(np.float32)
        comp_arr = np.array(source_fill.convert("RGB")).astype(np.float32)
        first_arr = np.array(first_pass.convert("RGB")).astype(np.float32)
        tex_arr = np.array(texture_layer.convert("RGB")).astype(np.float32)

        blur_r = max(1, int(feather_sigma * 6))
        src_soft = np.array(
            source_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32)[..., None] / 255.0
        tgt_soft = np.array(
            target_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32)[..., None] / 255.0

        w_tgt = tgt_soft
        w_src = src_soft * (1.0 - tgt_soft)
        w_bg = (1.0 - tgt_soft) * (1.0 - src_soft)

        target_px = texture_alpha * tex_arr + (1.0 - texture_alpha) * first_arr
        merged = target_px * w_tgt + comp_arr * w_src + orig_arr * w_bg
        return Image.fromarray(merged.clip(0, 255).astype(np.uint8))

    @staticmethod
    def _save_debug(debug_dir: str, name: str, img: Image.Image) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        img.save(os.path.join(debug_dir, name))

    @staticmethod
    def _mask_overlay(image: Image.Image, mask: Image.Image, color=(255, 0, 0, 100)) -> Image.Image:
        """Return image with the mask region tinted for visual inspection."""
        base = image.convert("RGBA")
        tint = Image.new("RGBA", base.size, color)
        alpha = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
        tint.putalpha(alpha)
        return Image.alpha_composite(base, tint).convert("RGB")

    def _cleanup_source_with_inpainting(
        self,
        image: Image.Image,
        source_mask: Image.Image,
        seed: int,
        num_inference_steps: int,
        debug_dir: str = None,
    ) -> Image.Image:
        """Fill the source hole using SD inpainting on the composite (Gaussian pre-fill).
        Empty positive prompt + strong animal negative keeps the model from regenerating the object.
        """
        pipe = self._load_inpaint_pipe()
        generator = self._make_generator(seed)
        mask = source_mask.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image_resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        mask_arr = np.array(mask)
        src_pixels = mask_arr > 127
        total = src_pixels.sum()
        h, w = mask_arr.shape
        ys, xs = np.where(src_pixels)

        print(f"[debug] source mask: {total} masked px  "
              f"bbox y=[{ys.min()}:{ys.max()}] x=[{xs.min()}:{xs.max()}]  "
              f"image size={image_resized.size}")

        inpaint_region = image_resized.crop((xs.min(), ys.min(), xs.max(), ys.max()))
        region_arr = np.array(inpaint_region)
        print(f"[debug] inpaint input region — "
              f"mean RGB=({region_arr[...,0].mean():.1f}, {region_arr[...,1].mean():.1f}, {region_arr[...,2].mean():.1f})  "
              f"std=({region_arr[...,0].std():.1f}, {region_arr[...,1].std():.1f}, {region_arr[...,2].std():.1f})")

        if debug_dir:
            self._save_debug(debug_dir, "1_inpaint_input.png", image_resized)
            self._save_debug(debug_dir, "2_inpaint_mask.png", mask)
            self._save_debug(debug_dir, "3_inpaint_input_overlay.png",
                             self._mask_overlay(image_resized, mask))
            self._save_debug(debug_dir, "4_inpaint_region_crop.png", inpaint_region)

        result = pipe(
            prompt="",
            negative_prompt=(
                "dog, cat, animal, pet, creature, fur, paw, tail, "
                "blurry, artifact, distorted, double exposure, ghost"
            ),
            image=image_resized,
            mask_image=mask,
            num_inference_steps=max(20, num_inference_steps),
            guidance_scale=7.5,
            strength=0.95,
            generator=generator,
        ).images[0]

        result_region = result.crop((xs.min(), ys.min(), xs.max(), ys.max()))
        result_arr = np.array(result_region)
        print(f"[debug] inpaint output region — "
              f"mean RGB=({result_arr[...,0].mean():.1f}, {result_arr[...,1].mean():.1f}, {result_arr[...,2].mean():.1f})  "
              f"std=({result_arr[...,0].std():.1f}, {result_arr[...,1].std():.1f}, {result_arr[...,2].std():.1f})")

        if debug_dir:
            self._save_debug(debug_dir, "5_inpaint_output.png", result)
            self._save_debug(debug_dir, "6_inpaint_output_overlay.png",
                             self._mask_overlay(result, mask))
            self._save_debug(debug_dir, "7_inpaint_output_region_crop.png", result_region)

        return result

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0].float()

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev_int: int,
        eps_pred: torch.Tensor,
        stored_noise: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM reverse step, injecting stored (optionally shifted) noise."""
        t_int = t.item()
        abar_t = self.scheduler.alphas_cumprod[t_int].to(device=self.device, dtype=torch.float32)

        # SD 2.1 v-prediction → convert to noise prediction
        if self.prediction_type == "v_prediction":
            eps_pred = abar_t.sqrt() * eps_pred + (1 - abar_t).sqrt() * x_t

        pred_x0 = (x_t - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
        pred_x0 = pred_x0.clamp(-4.0, 4.0)

        if t_prev_int < 0:
            return pred_x0

        abar_prev = self.scheduler.alphas_cumprod[t_prev_int].to(device=self.device, dtype=torch.float32)
        coeff1 = abar_prev.sqrt() * (1 - abar_t / abar_prev) / (1 - abar_t)
        coeff2 = (abar_t / abar_prev).sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * pred_x0 + coeff2 * x_t

        beta_t = (1 - abar_t / abar_prev).clamp(min=0.0)
        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
        return mu + sigma_t * stored_noise

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        target_mask: Image.Image,
        use_noise_shift: bool = True,
        seed: int = 42,
        num_inference_steps: int = 50,
        sdedit_strength: float = 0.7,
        guidance_scale: float = 7.5,
        feather_sigma: float = 2.0,
        debug_dir: str = None,
    ) -> Image.Image:
        """
        Move the object defined by source_mask to target_mask.

        Pipeline:
          1. Pixel copy-paste → composite (establishes WHERE the object goes)
          2. DDPM inversion of original → a reconstruction-consistent latent trajectory
          3. (ours) Shift the per-step noise maps source→target
          4. SDEdit from composite with RePaint-style locks for background/source cleanup

        use_noise_shift=True → ours; False → baseline without spatial noise shifting.
        """
        device = self.device
        sz = self.image_size

        # 0. Fast path: if nothing moves, preserve the input exactly.
        if self._masks_are_effectively_same(source_mask, target_mask):
            identity = image.convert("RGB").resize((self.image_size, self.image_size))
            return identity.copy(), identity

        # 1. Pixel-space copy-paste
        img_sz = image.resize((sz, sz))
        composite = create_composite(
            img_sz,
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
        )
        if debug_dir:
            src_mask_sz = source_mask.resize((sz, sz))
            tgt_mask_sz = target_mask.resize((sz, sz))
            self._save_debug(debug_dir, "0a_original_resized.png", img_sz)
            self._save_debug(debug_dir, "0b_composite_gaussian_fill.png", composite)
            self._save_debug(debug_dir, "0c_source_mask.png", src_mask_sz.convert("L"))
            self._save_debug(debug_dir, "0d_target_mask.png", tgt_mask_sz.convert("L"))
            self._save_debug(debug_dir, "0e_composite_src_overlay.png",
                             self._mask_overlay(composite, src_mask_sz.convert("L")))
            src_arr = np.array(src_mask_sz.convert("L")) > 127
            region = np.array(composite)[src_arr]
            print(f"[debug] Gaussian fill at source — "
                  f"mean RGB=({region[:,0].mean():.1f}, {region[:,1].mean():.1f}, {region[:,2].mean():.1f})  "
                  f"std=({region[:,0].std():.1f}, {region[:,1].std():.1f}, {region[:,2].std():.1f})")
            blur_ys, blur_xs = np.where(src_arr)
            box_h = int(blur_ys.max() - blur_ys.min() + 1)
            box_w = int(blur_xs.max() - blur_xs.min() + 1)
            print(f"[debug] source bbox: {box_h}×{box_w}  blur_r used: {max(4, max(box_h, box_w))}")

        # 2. Encode original + composite
        vae_dtype = next(self.vae.parameters()).dtype
        x0_orig = encode_image(self.vae, pil_to_tensor(img_sz, device).to(vae_dtype))
        x0_composite = encode_image(self.vae, pil_to_tensor(composite, device).to(vae_dtype))

        # 3. Encode prompts for CFG
        encoder_hs = self._encode_prompt(prompt)
        uncond_hs = self._encode_prompt("")

        # 4. Prepare latent masks and soft locks
        M_src = prepare_latent_mask(source_mask.resize((sz, sz)), device, self.latent_size)
        M_tgt = prepare_latent_mask(target_mask.resize((sz, sz)), device, self.latent_size)
        M_src_soft = gaussian_blur_mask(M_src, sigma=2.0)
        M_tgt_soft = gaussian_blur_mask(M_tgt, sigma=2.0)
        bg_mask = (1.0 - M_src_soft.clamp(0, 1)) * (1.0 - M_tgt_soft.clamp(0, 1))
        M_tgt_lock = gaussian_blur_mask(M_tgt, sigma=2.0)

        # 5. Invert the original once, then optionally shift the stored noise maps
        inversion = ddpm_invert(x0_orig, self.scheduler, num_inference_steps, seed, device)

        if use_noise_shift:
            start_noises = shift_all_noise_maps(
                inversion.marginal_noises, M_src, M_tgt, device, feather_sigma
            )
            transition_noises = shift_all_noise_maps(
                inversion.transition_noises, M_src, M_tgt, device, feather_sigma
            )
        else:
            start_noises = inversion.marginal_noises
            transition_noises = inversion.transition_noises

        # 6. SDEdit start from the composite at the chosen timestep
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        start_idx = max(0, min(int((1.0 - sdedit_strength) * len(timesteps)), len(timesteps) - 1))
        t_start_int = timesteps[start_idx].item()
        abar_start = self.scheduler.alphas_cumprod[t_start_int].to(device=device, dtype=torch.float32)
        start_noise = start_noises[t_start_int]
        x_t = abar_start.sqrt() * x0_composite + (1 - abar_start).sqrt() * start_noise

        # 7. Denoising loop with background and early target locking
        unet_dtype = next(self.unet.parameters()).dtype
        active_timesteps = timesteps[start_idx:]
        lock_cutoff = len(active_timesteps) // 4

        for i, t in enumerate(active_timesteps):
            t_global_idx = start_idx + i
            t_prev_int = timesteps[t_global_idx + 1].item() if t_global_idx + 1 < len(timesteps) else -1

            with torch.no_grad():
                x_t_input = x_t.to(unet_dtype)
                t_batch = t.unsqueeze(0).to(device)
                latent_batch = torch.cat([x_t_input, x_t_input], dim=0)
                cond_batch = torch.cat([uncond_hs, encoder_hs], dim=0).to(unet_dtype)
                noise_pred = self.unet(latent_batch, t_batch.repeat(2), cond_batch).sample.float()
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            stored = transition_noises.get(t_prev_int, torch.zeros_like(x_t))
            x_t = self._ddpm_step(x_t, t, t_prev_int, eps_pred, stored)

            if t_prev_int >= 0:
                x_t_orig = inversion.latents[t_prev_int]
                x_t = x_t * (1.0 - bg_mask) + x_t_orig * bg_mask

                if i < lock_cutoff:
                    x_t_comp = reconstruct_xt(
                        x0_composite,
                        t_prev_int,
                        start_noises[t_prev_int],
                        self.scheduler,
                        device,
                    )
                    x_t = x_t * (1.0 - M_tgt_lock) + x_t_comp * M_tgt_lock

        # 8. Decode the SDEdit result.
        decoded = decode_latent(self.vae, x_t.to(vae_dtype))
        first_pass = tensor_to_pil(decoded)
        if debug_dir:
            self._save_debug(debug_dir, "8_sdedit_first_pass.png", first_pass)

        # 9. Fill the source hole: run SD inpainting on the composite (Gaussian pre-fill, no dog)
        #    with an empty prompt + strong animal negative so it fills with background texture.
        cleaned = self._cleanup_source_with_inpainting(
            composite,
            source_mask.resize((sz, sz)),
            seed + 1,
            num_inference_steps,
            debug_dir=debug_dir,
        )
        if debug_dir:
            self._save_debug(debug_dir, "9_cleaned_source.png", cleaned)

        # 10. Texture transplant: shift original object pixels to target position for sharpness.
        texture_layer = self._shift_pixels_to_target(
            img_sz,
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
        )
        if debug_dir:
            self._save_debug(debug_dir, "10_texture_layer.png", texture_layer)

        # 11. Final composite: sharp target (texture + SDEdit blend), inpainted source fill, original bg.
        final = self._compose_final(
            img_sz,
            cleaned,
            first_pass,
            texture_layer,
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
            feather_sigma,
        )
        if debug_dir:
            self._save_debug(debug_dir, "11_final.png", final)
            print(f"[debug] all debug images saved to: {debug_dir}")
        return final, composite
