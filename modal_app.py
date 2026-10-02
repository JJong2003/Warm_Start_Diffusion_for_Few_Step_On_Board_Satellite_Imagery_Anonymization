"""Modal 서버리스 GPU 엔드포인트.

orchestrator/pipeline.py (탐지 -> 마스크 -> coarse-fill -> warm-start diffusion -> 합성)를
그대로 import해서 Modal GPU 컨테이너에서 돌린다. Spring(OrchestratorWarmStartProcessor)이
기대하는 POST /infer 응답 형식(5단계 이미지 + meta)은 orchestrator/server.py 와 동일하다.

가중치:
  - YOLO / LoRA : Modal Volume "satellite-anon-weights" (modal volume put 으로 업로드)
  - SD1.5 inpaint base : 이미지 빌드 시점에 HuggingFace에서 받아 이미지에 포함

실행:
  modal serve modal_app.py     # 임시 URL로 테스트 (저장하면 자동 반영)
  modal deploy modal_app.py    # 영구 URL 발급
"""
import base64
import os

import modal

APP_NAME = "satellite-anonymizer"
VOLUME_NAME = "satellite-anon-weights"
SECRET_NAME = "orchestrator-api-key"  # ORCH_API_KEY 하나만 들어있는 Secret

BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BASE_DIR = "/models/sd-inpaint"
WEIGHTS_DIR = "/weights"
DET_WEIGHTS = f"{WEIGHTS_DIR}/yolo_obb_dota/best.pt"
LORA_DIR = f"{WEIGHTS_DIR}/inpaint_lora"


def _download_base_model():
    """이미지 빌드 시 1회 실행. pipeline.py가 variant 없이 로드하므로 fp32 .bin 파일을 받는다."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        BASE_REPO,
        local_dir=BASE_DIR,
        allow_patterns=[
            "model_index.json",
            "scheduler/*.json",
            "tokenizer/*",
            "*/config.json",
            "unet/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.bin",
            "text_encoder/pytorch_model.bin",
        ],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # ultralytics가 끌어오는 opencv용
    .pip_install(
        "torch",
        "diffusers==0.40.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "ultralytics==8.4.157",
        "accelerate",
        "huggingface_hub",
        "fastapi[standard]",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics"})
    .run_function(_download_base_model)
    .add_local_file("orchestrator/pipeline.py", "/root/pipeline.py")
)

app = modal.App(APP_NAME)
weights = modal.Volume.from_name(VOLUME_NAME)


@app.cls(
    image=image,
    gpu="T4",
    volumes={WEIGHTS_DIR: weights},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=300,  # 마지막 요청 후 5분간 컨테이너 유지 (콜드스타트 완화)
    timeout=900,
)
class Anonymizer:
    @modal.enter()
    def load(self):
        from pipeline import Pipeline

        print(f">> 모델 로딩 (det={DET_WEIGHTS}, lora={LORA_DIR})")
        self.pipeline = Pipeline(DET_WEIGHTS, BASE_DIR, LORA_DIR, "cuda")
        print(">> 로딩 완료")

    @modal.asgi_app()
    def web(self):
        import cv2
        import numpy as np
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import JSONResponse

        api_key = os.environ.get("ORCH_API_KEY", "")
        web_app = FastAPI(title="model-showcase orchestrator (modal)")

        def to_data_url(img) -> str:
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("이미지 인코딩 실패")
            return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

        @web_app.get("/health")
        def health():
            return {"status": "ok", "device": "cuda", "ready": True}

        @web_app.post("/infer")
        async def infer(
            image: UploadFile = File(...),
            model_id: str = Form("warm-start-anonymizer"),
            x_api_key: str = Header(default=""),
        ):
            if api_key and x_api_key != api_key:
                raise HTTPException(status_code=401, detail="invalid api key")
            if model_id != "warm-start-anonymizer":
                raise HTTPException(status_code=400, detail=f"지원하지 않는 model_id: {model_id}")

            raw = await image.read()
            img_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

            result = self.pipeline.process(img_bgr)
            return JSONResponse({
                "boxes_image": to_data_url(result["boxes_image"]),
                "mask_image": to_data_url(result["mask_image"]),
                "coarse_fill_image": to_data_url(result["coarse_fill_image"]),
                "diffusion_preview_image": to_data_url(result["diffusion_preview_image"]),
                "result_image": to_data_url(result["result_image"]),
                "meta": result["meta"],
            })

        return web_app
