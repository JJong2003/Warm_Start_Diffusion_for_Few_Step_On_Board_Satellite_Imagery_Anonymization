"""내부 GPU 오케스트레이터 (FastAPI).

Spring 백엔드(JobQueueService)가 job 하나를 워커 스레드에서 순차 처리하면서
이 서버의 POST /infer 를 동기 호출한다 — 그래서 이 서버 자체는 큐를 두지 않고
요청 하나를 바로 처리해서 돌려주면 된다 (전역 큐는 Spring 쪽에 이미 있음).

로컬 테스트: uvicorn server:app --host 0.0.0.0 --port 8000
서버 컴퓨터로 옮길 때는 DET_WEIGHTS/BASE_MODEL_DIR/LORA_DIR/DEVICE 환경변수만
그 장비 경로/GPU에 맞게 바꾸고, Spring 쪽 app.orchestrator.base-url을 그 주소로 바꾸면 된다.
"""
import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "모델" / "satellite"

DET_WEIGHTS = os.environ.get("DET_WEIGHTS", str(MODEL_ROOT / "yolo_obb_dota" / "best.pt"))
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", str(MODEL_ROOT / "sd-inpaint"))
LORA_DIR = os.environ.get("LORA_DIR", str(MODEL_ROOT / "inpaint_lora"))
DEVICE = os.environ.get("DEVICE", "cpu")

app = FastAPI(title="model-showcase orchestrator")
pipeline: Pipeline | None = None


@app.on_event("startup")
def load_models():
    global pipeline
    print(f">> 모델 로딩 중 (device={DEVICE})")
    print(f"   det   = {DET_WEIGHTS}")
    print(f"   base  = {BASE_MODEL_DIR}")
    print(f"   lora  = {LORA_DIR}")
    pipeline = Pipeline(DET_WEIGHTS, BASE_MODEL_DIR, LORA_DIR, DEVICE)
    print(">> 로딩 완료, 요청 대기 중")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "ready": pipeline is not None}


def _to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("이미지 인코딩 실패")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/infer")
async def infer(image: UploadFile = File(...), model_id: str = Form("warm-start-anonymizer")):
    if model_id != "warm-start-anonymizer":
        raise HTTPException(status_code=400, detail=f"지원하지 않는 model_id: {model_id}")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="모델을 아직 불러오는 중입니다.")

    raw = await image.read()
    img_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

    result = pipeline.process(img_bgr)

    return JSONResponse({
        "boxes_image": _to_data_url(result["boxes_image"]),
        "mask_image": _to_data_url(result["mask_image"]),
        "coarse_fill_image": _to_data_url(result["coarse_fill_image"]),
        "diffusion_preview_image": _to_data_url(result["diffusion_preview_image"]),
        "result_image": _to_data_url(result["result_image"]),
        "meta": result["meta"],
    })
