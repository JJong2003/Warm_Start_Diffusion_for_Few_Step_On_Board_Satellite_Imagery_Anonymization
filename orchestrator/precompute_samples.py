"""샘플 갤러리 이미지에 대해 파이프라인을 한 번씩 미리 돌려 5단계 결과를 저장한다.

server.py(FastAPI)는 거치지 않고 pipeline.py의 Pipeline을 직접 사용한다.
샘플 이미지를 교체하거나 모델/하이퍼파라미터가 바뀔 때만 수동으로 재실행하면 된다.

실행:
    .venv/Scripts/python.exe orchestrator/precompute_samples.py
"""
import json
import time
from pathlib import Path

import cv2

from pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "모델" / "satellite"
STATIC_ROOT = REPO_ROOT / "src" / "main" / "resources" / "static"

DET_WEIGHTS = str(MODEL_ROOT / "yolo_obb_dota" / "best.pt")
BASE_MODEL_DIR = str(MODEL_ROOT / "sd-inpaint")
LORA_DIR = str(MODEL_ROOT / "inpaint_lora")
DEVICE = "cpu"

SAMPLES = [
    {"id": "airport-apron", "src": STATIC_ROOT / "img" / "samples" / "airport-apron.jpg"},
    {"id": "harbor-ships-candidate", "src": STATIC_ROOT / "img" / "samples" / "harbor-ships-candidate.png"},
    {"id": "helicopter-apron", "src": STATIC_ROOT / "img" / "samples" / "helicopter-apron.png"},
]

OUT_ROOT = STATIC_ROOT / "img" / "precomputed"
# Spring 클래스패스 리소스로 바로 저장 — PrecomputedSampleStore가 부팅 시 읽어들인다.
# (재생성 후에는 Spring Boot 앱을 다시 빌드/기동해야 반영됨)
MANIFEST_PATH = REPO_ROOT / "src" / "main" / "resources" / "precomputed" / "samples.json"

STAGE_FILES = {
    "boxes_image": "boxes.png",
    "mask_image": "mask.png",
    "coarse_fill_image": "coarse_fill.png",
    "diffusion_preview_image": "diffusion_preview.png",
    "result_image": "result.png",
}


def main():
    print(f">> 모델 로딩 중 (device={DEVICE})")
    print(f"   det  = {DET_WEIGHTS}")
    print(f"   base = {BASE_MODEL_DIR}")
    print(f"   lora = {LORA_DIR}")
    t_load = time.time()
    pipeline = Pipeline(DET_WEIGHTS, BASE_MODEL_DIR, LORA_DIR, DEVICE)
    print(f">> 로딩 완료 ({time.time() - t_load:.1f}s)")

    manifest = {}
    for sample in SAMPLES:
        sid, src = sample["id"], sample["src"]
        print(f"\n>> [{sid}] 처리 중: {src}")
        if not src.exists():
            print(f"   !! 파일 없음: {src}")
            continue
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            print(f"   !! 이미지를 읽을 수 없음: {src}")
            continue

        t0 = time.time()
        result = pipeline.process(img_bgr)
        elapsed = time.time() - t0
        print(f"   완료: {elapsed:.1f}s, meta={result['meta']}")

        out_dir = OUT_ROOT / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        urls = {}
        for key, filename in STAGE_FILES.items():
            ok = cv2.imwrite(str(out_dir / filename), result[key])
            if not ok:
                print(f"   !! 저장 실패: {filename}")
            urls[key] = f"/img/precomputed/{sid}/{filename}"

        manifest[sid] = {**urls, "meta": result["meta"]}

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n>> 매니페스트 저장: {MANIFEST_PATH}")
    print(">> 전체 완료")


if __name__ == "__main__":
    main()
