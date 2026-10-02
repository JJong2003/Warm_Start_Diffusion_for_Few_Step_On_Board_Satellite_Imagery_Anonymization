---
title: Satellite Imagery Anonymization Demo
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: creativeml-openrail-m
---

# 위성 영상 익명화 데모

*Warm-Start Diffusion for Few-Step On-Board Satellite Imagery Anonymization* 논문 모델을
누구나 웹에서 체험해볼 수 있게 만든 데모입니다.

1. YOLO11s-OBB로 민감 시설(plane/ship/helicopter) 탐지
2. 탐지 박스를 팽창해 인페인팅 마스크 생성
3. OpenCV Telea 알고리즘으로 coarse-fill 생성
4. coarse-fill을 latent로 인코딩 후 warm-start + SD1.5 Inpainting(LoRA) + DPM-Solver++ 10 NFE로 복원

무료 ZeroGPU 쿼터로 서빙되므로 일일 사용량 제한이 있습니다.
