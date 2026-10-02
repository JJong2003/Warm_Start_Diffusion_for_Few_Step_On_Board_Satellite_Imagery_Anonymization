# Model Showcase — 위성 영상 익명화 모델 체험 플랫폼

논문 *Warm-Start Diffusion for Few-Step On-Board Satellite Imagery Anonymization*(Baek, Han, Yun, Park, Koh — 충남대학교)에서 제안한 위성 영상 익명화 모델을, 누구나 웹에서 직접 체험해볼 수 있게 만든 데모 플랫폼입니다.

이미지를 업로드하면 탐지 → 마스크 생성 → coarse-fill → diffusion 복원 → 합성으로 이어지는 파이프라인이 단계별로 시각화되어, 수식이나 학습 과정 없이도 "위성 사진 속 민감 시설이 어떻게 지워지는지"를 눈으로 확인할 수 있습니다.

## 링크

| 구분 | 주소 |
|---|---|
| 체험 사이트 (모델 직접 사용해보기) | https://warm-start-diffusion-for-few-step-on.onrender.com/ |
| GitHub | https://github.com/JJong2003/Warm_Start_Diffusion_for_Few_Step_On_Board_Satellite_Imagery_Anonymization |
| 논문 원문 | _아직 미게시 — 게시되면 여기에 기록_ |

사이트 상단바·푸터에 표시되는 링크는 [`src/main/resources/static/js/main.js`](src/main/resources/static/js/main.js) 맨 위의 `LINKS` 객체에서 관리합니다. 논문이 게시되면 `LINKS.paper`에 URL을 넣으면 "준비 중" 표시가 사라지고 링크가 활성화됩니다. (위 표도 함께 갱신해 주세요.)

## 모델 소개

위성 사진이 지상으로 전송되기 전, 온보드(on-board) 환경에서 민감 시설(전투기, 군항 등)을 자동으로 탐지해 지우는 파이프라인입니다.

1. **탐지**: YOLO11s-OBB로 민감 객체의 방향성 바운딩 박스(OBB) 탐지
2. **마스크 생성**: 탐지 박스를 9픽셀 팽창해 인페인팅 마스크 생성
3. **Coarse-fill**: OpenCV Telea 알고리즘으로 배경을 거칠게 채워 밑그림 생성
4. **Diffusion 복원**: coarse-fill을 latent로 인코딩하고 약한 노이즈를 더한 뒤(warm-start), SD1.5 Inpainting(LoRA 파인튜닝) + DPM-Solver++로 단 10 NFE 만에 복원
5. **합성**: 복원된 영역을 원본 이미지와 합성

이 warm-start 방식 덕분에 적은 추론 스텝만으로도 빠르게 결과를 얻을 수 있습니다. 논문에서 보고한 성능은 DOTA v1.0 기준 탐지 회피율 97.58%, LPIPS 0.1096이며, 데스크톱 GPU(RTX 5080) 기준 298ms / 약 3GB VRAM이 소요됩니다. 최종 목표는 Jetson Orin Nano(8GB)급 온보드 기기에서의 실시간 처리입니다.

## 이 프로젝트가 제공하는 것

- **이미지 입력**: 직접 업로드하거나 준비된 샘플 이미지 중 선택 (샘플 이미지는 실제 위성 사진이 아니라 AI가 생성한 예시입니다)
- **파이프라인 시각화**: 원본을 포함한 5단계(탐지 → 마스크 → coarse-fill → diffusion → 합성)를 애니메이션/스텝 형태로 노출
- **결과 비교**: 원본과 결과 이미지를 슬라이더로 비교하고 다운로드

## 아키텍처

```
[사용자 브라우저]
      │ HTTPS
[Spring Boot 게이트웨이]   정적 프런트엔드 서빙 · 업로드 검증 · 속도 제한 · 작업 제출/상태 폴링 API
      │ 내부망 (이미지)
[Python GPU 오케스트레이터]   전역 큐 · 추론 워커
```

- 공개 API 서버 역할은 Spring Boot(Java 21)가 담당하고, 실제 모델 추론(PyTorch / diffusers / ultralytics)은 Python 오케스트레이터가 담당합니다.
- 계정이나 이력을 저장하지 않는 무상태 서비스로, 별도 DB 없이 업로드 → 처리 → 응답 흐름만 처리합니다.
- 클라이언트 사이드 캔버스로 동작을 흉내 내는 데모 모드를 지원해, 백엔드 없이도 UI를 확인할 수 있습니다.
