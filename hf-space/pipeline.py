"""Warm-start diffusion 위성 영상 익명화 파이프라인.

orchestrator/pipeline.py 와 동일한 로직(논문 재현값 그대로). ZeroGPU Space에서는
Pipeline 클래스 대신 app.py가 detector/remover를 직접 들고 있으므로, 여기서는
WarmStartRemover와 마스크/박스 유틸리티만 가져다 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                        UNet2DConditionModel)
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer

PROMPT = "aerial satellite view, empty ground, natural terrain"
STRENGTH = 0.85      # warm-start 강도 (s)
DPM_STEPS = 10        # NFE
DILATE = 9            # 추론 마스크 dilation (px)
TARGET_CLASSES = {"plane", "ship", "helicopter"}  # LoRA가 비행기/배경 도메인 위주로 학습됨 — ship/helicopter는 인페인팅 품질이 상대적으로 떨어질 수 있음

BOX_COLOR_BGR = (79, 102, 232)


class WarmStartRemover:
    def __init__(self, base: str, lora: str, device: str, size: int = 512, fp16: bool = True):
        self.device, self.size = device, size
        dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.dtype = dtype

        self.vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae").to(device).eval().to(dtype)
        self.txt = CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder").to(device).eval().to(dtype)
        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.dpm = DPMSolverMultistepScheduler.from_pretrained(
            base, subfolder="scheduler",
            algorithm_type="dpmsolver++", solver_order=2)

        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(device)
        self.unet = PeftModel.from_pretrained(
            unet, lora).merge_and_unload().eval().to(dtype)

        self.sf = self.vae.config.scaling_factor
        ids = tok(PROMPT, padding="max_length",
                  max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            self.ctx = self.txt(ids)[0]

    def _to_tensor(self, rgb):
        im = cv2.resize(rgb, (self.size, self.size))
        return (torch.from_numpy(im).float().permute(2, 0, 1)[None]
                .to(self.device) / 127.5 - 1.0).to(self.dtype)

    def coarse_fill(self, img_bgr, mask, down=4):
        """저해상도 Telea 인페인팅으로 빠른 coarse-fill."""
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (max(1, w // down), max(1, h // down)))
        m_small = cv2.resize(mask, (max(1, w // down), max(1, h // down)),
                             interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, m_small, 5, cv2.INPAINT_TELEA)
        return cv2.resize(filled, (w, h))

    @torch.no_grad()
    def run(self, img_bgr, mask, coarse=None):
        dev = self.device
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_bgr.shape[:2]

        x = self._to_tensor(rgb)
        ms = cv2.resize(mask, (self.size, self.size),
                        interpolation=cv2.INTER_NEAREST)
        m = ((torch.from_numpy(ms).float()[None, None].to(dev) / 255.0) > 0.5
             ).to(self.dtype)

        # 조건화: masked latent (마스크 안쪽 0) — 학습과 동일하게 zero-fill
        z_masked = self.vae.encode(x * (1 - m)).latent_dist.sample() * self.sf
        m_lat = F.interpolate(m.float(), size=z_masked.shape[-2:],
                              mode="nearest").to(self.dtype)

        if coarse is None:
            coarse = self.coarse_fill(img_bgr, mask)
        z_c = self.vae.encode(
            self._to_tensor(cv2.cvtColor(coarse, cv2.COLOR_BGR2RGB))
        ).latent_dist.sample() * self.sf

        full = max(int(round(DPM_STEPS / STRENGTH)), DPM_STEPS + 1)
        self.dpm.set_timesteps(full, device=dev)
        ts = self.dpm.timesteps[full - DPM_STEPS:]
        self.dpm._step_index = None
        z = self.dpm.add_noise(z_c, torch.randn_like(z_c), ts[0].reshape(1))

        for t in ts:
            eps = self.unet(torch.cat([z, m_lat, z_masked], dim=1),
                            t, encoder_hidden_states=self.ctx).sample
            z = self.dpm.step(eps, t, z).prev_sample

        out = self.vae.decode(z / self.sf).sample
        out = ((out.float().clamp(-1, 1)[0] + 1) * 127.5
               ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        out = cv2.resize(out, (W0, H0))

        keep = (mask[..., None] > 127).astype(np.uint8)
        rgb_out = out * keep + rgb * (1 - keep)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


def build_mask(res, names, want, H, W, dilate=DILATE):
    """YOLO-OBB 결과 -> 이진 마스크 + 탐지 폴리곤 목록."""
    m = np.zeros((H, W), np.uint8)
    polys = []
    if res.obb is not None and res.obb.xyxyxyxy is not None:
        for poly, c, conf in zip(res.obb.xyxyxyxy.cpu().numpy(),
                                  res.obb.cls.cpu().numpy().astype(int),
                                  res.obb.conf.cpu().numpy()):
            if names[c] in want:
                cv2.fillPoly(m, [poly.astype(np.int32)], 255)
                polys.append((poly.astype(np.int32), names[c], float(conf)))
    if polys:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
    return m, polys


def draw_boxes(img_bgr, polys):
    out = img_bgr.copy()
    thickness = max(2, out.shape[1] // 300)
    for poly, cls_name, conf in polys:
        cv2.polylines(out, [poly], isClosed=True, color=BOX_COLOR_BGR, thickness=thickness)
        x, y = poly[0]
        cv2.putText(out, f"{cls_name} {conf:.2f}", (int(x), max(0, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR_BGR, 1, cv2.LINE_AA)
    return out
