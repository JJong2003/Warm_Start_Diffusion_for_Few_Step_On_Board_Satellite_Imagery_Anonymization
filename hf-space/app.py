"""ZeroGPU Space: 위성 영상 익명화 데모 (Warm-Start Diffusion).

로컬의 Spring Boot 게이트웨이 + FastAPI 오케스트레이터 구조 대신, HF Spaces의
무료 ZeroGPU 쿼터로 호스팅하기 위해 단일 Gradio 앱으로 파이프라인을 실행한다.
ZeroGPU는 Gradio SDK에서만 동작하고, 실제 GPU는 @spaces.GPU로 감싼 함수 안에서만
붙기 때문에 요청당 무거운 연산(탐지+diffusion)을 전부 그 함수 안에서 수행한다.

무료 계정 쿼터는 하루 5분이고, 실제 걸린 시간이 아니라 duration에 선언한 값만큼
차감되므로 duration은 넉넉하되 과하지 않게 잡는다 (첫 호출은 CUDA 커널 warm-up이
끼어서 이후 호출보다 느리다).
"""
import os

import cv2
import gradio as gr
import numpy as np
import spaces
from ultralytics import YOLO

from pipeline import DPM_STEPS, TARGET_CLASSES, WarmStartRemover, build_mask, draw_boxes

BASE_MODEL = os.environ.get("BASE_MODEL_DIR", "stable-diffusion-v1-5/stable-diffusion-inpainting")
LORA_DIR = os.environ.get("LORA_DIR", "weights/inpaint_lora")
DET_WEIGHTS = os.environ.get("DET_WEIGHTS", "weights/yolo_obb_dota/best.pt")

# diffusion 모델은 모듈 최상단에서 미리 "cuda"로 올려둔다 (ZeroGPU 권장 방식 —
# @spaces.GPU 밖에서는 CUDA 호출이 에뮬레이션되므로 여기서의 .to("cuda")는 안전하고,
# 요청마다 다시 옮기는 것보다 훨씬 효율적이다).
remover = WarmStartRemover(BASE_MODEL, LORA_DIR, device="cuda")

# YOLO는 작고(약 20MB) ultralytics가 호출 시점에 device를 받으므로 CPU에 둔 채
# 추론 시에만 device="cuda"를 넘긴다.
detector = YOLO(DET_WEIGHTS)


@spaces.GPU(duration=20)
def run_pipeline(image_rgb: np.ndarray, conf: float):
    if image_rgb is None:
        raise gr.Error("이미지를 업로드해주세요.")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]

    res = detector(image_bgr, conf=conf, device="cuda", verbose=False)[0]
    mask, polys = build_mask(res, detector.names, TARGET_CLASSES, H, W)
    boxes_image = cv2.cvtColor(draw_boxes(image_bgr, polys), cv2.COLOR_BGR2RGB)

    if not polys:
        empty_mask = np.zeros((H, W), np.uint8)
        status = "민감 시설(plane/ship/helicopter)이 탐지되지 않았습니다."
        return boxes_image, empty_mask, image_rgb, image_rgb, status

    coarse = remover.coarse_fill(image_bgr, mask)
    result_bgr = remover.run(image_bgr, mask, coarse=coarse)

    coarse_rgb = cv2.cvtColor(coarse, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    status = f"탐지 {len(polys)}건 · {DPM_STEPS} NFE로 복원 완료"
    return boxes_image, mask, coarse_rgb, result_rgb, status


with gr.Blocks(title="위성 영상 익명화 데모") as demo:
    gr.Markdown(
        "# 위성 영상 익명화 데모\n"
        "*Warm-Start Diffusion for Few-Step On-Board Satellite Imagery Anonymization* 논문 모델 체험\n\n"
        "이미지를 업로드하면 **탐지 → 마스크 → coarse-fill → diffusion 복원** 순서로 처리됩니다. "
        "무료 ZeroGPU 쿼터로 서빙 중이라 혼잡하거나 일일 한도를 넘으면 잠시 대기하거나 다음 날 다시 시도해주세요."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="원본 위성 이미지")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="탐지 confidence threshold")
            run_btn = gr.Button("실행", variant="primary")
            status = gr.Textbox(label="상태", interactive=False)
        with gr.Column(scale=2):
            with gr.Row():
                out_boxes = gr.Image(label="1. 탐지 (YOLO11s-OBB)")
                out_mask = gr.Image(label="2. 마스크")
            with gr.Row():
                out_coarse = gr.Image(label="3. Coarse-fill")
                out_result = gr.Image(label="4. Diffusion 복원 결과")

    run_btn.click(
        run_pipeline,
        inputs=[inp, conf],
        outputs=[out_boxes, out_mask, out_coarse, out_result, status],
    )

if __name__ == "__main__":
    demo.launch()
