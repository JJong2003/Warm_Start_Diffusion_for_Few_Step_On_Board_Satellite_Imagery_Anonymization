(function(){
  "use strict";

  // ============================================================
  // 이 프론트엔드는 Spring 앱 자신이 서빙하므로 기본은 같은 origin(/api/jobs)을 쓴다.
  // 프론트엔드만 별도 호스팅(GitHub Pages 등)할 때는 아래 값을 그 백엔드 주소로 바꾸면 된다.
  // ============================================================
  const DEFAULT_API_BASE_URL = window.location.origin; // 예: "https://your-gpu-server.example.com"

  // ============================================================
  // 외부 링크 (상단바·푸터의 [data-link] 요소에 자동 반영된다).
  // 논문이 게시되면 paper에 URL만 넣으면 "준비 중" 표시가 사라지고 링크가 활성화된다.
  // ============================================================
  const LINKS = {
    github: 'https://github.com/JJong2003/Warm_Start_Diffusion_for_Few_Step_On_Board_Satellite_Imagery_Anonymization',
    paper: '', // 예: "https://arxiv.org/abs/XXXX.XXXXX" — 아직 미게시
  };

  let apiBaseUrl = DEFAULT_API_BASE_URL.trim();
  let selected = null;      // { kind:'upload'|'sample', getBlob:()=>Promise<Blob>, previewUrl, label, width, height }
  let stages = [];          // rendered stage objects: {code,name,desc,imgUrl}
  let currentStageIdx = 0;
  let isProcessing = false;

  const STAGE_DEFS = [
    { key:'input',   code:'00', name:'원본',    desc:'업로드된 원본 위성 사진입니다.' },
    { key:'detect',  code:'01', name:'DETECT',  desc:'YOLO-OBB가 민감한 대상의 위치와 방향을 회전된 사각형으로 찾아냅니다.' },
    { key:'mask',    code:'02', name:'MASK',    desc:'탐지된 영역을 여유 있게 감싸는 제거 마스크를 만듭니다.' },
    { key:'fill',    code:'03', name:'FILL',    desc:'마스크 안쪽을 주변 배경 색과 구조로 거칠게 채워 밑그림을 만듭니다.' },
    { key:'diffuse', code:'04', name:'DIFFUSE', desc:'밑그림에 약한 노이즈를 더한 뒤, 10단계 확산 모델(DPM-Solver++)로 자연스럽게 복원합니다.' },
    { key:'output',  code:'05', name:'OUTPUT',  desc:'복원된 영역을 원본과 합성한 최종 익명화 이미지입니다.' },
  ];

  // ---------- DOM refs ----------
  const $ = (id) => document.getElementById(id);
  const dropzone = $('dropzone'), fileInput = $('fileInput');
  const previewThumb = $('previewThumb'), previewMeta = $('previewMeta');
  const sampleRow = $('sampleRow');
  const runBtn = $('runBtn'), runMsg = $('runMsg');
  const stageListEl = $('stageList');
  const viewerCode = $('viewerCode'), viewerName = $('viewerName'), demoBadge = $('demoBadge');
  const viewerBody = $('viewerBody'), viewerCaption = $('viewerCaption');
  const prevBtn = $('prevBtn'), nextBtn = $('nextBtn');
  const compareSection = $('compareSection'), compareBefore = $('compareBefore'), compareAfter = $('compareAfter');
  const compareAfterWrap = $('compareAfterWrap'), compareHandle = $('compareHandle'), compareSlider = $('compareSlider');
  const downloadBtn = $('downloadBtn');
  const apiUrlInput = $('apiUrlInput'), saveUrlBtn = $('saveUrlBtn'), testUrlBtn = $('testUrlBtn'), advStatus = $('advStatus');

  // ---------- helpers ----------
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // ---------- external links ----------
  document.querySelectorAll('[data-link]').forEach((a) => {
    const url = LINKS[a.dataset.link];
    if (!url) return; // 값이 없으면 HTML 기본값(준비 중 표시)을 그대로 둔다
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.classList.remove('is-pending');
    const tag = a.querySelector('.pending-tag');
    if (tag) tag.remove();
  });

  // ---------- theme toggle (dark / light) ----------
  const themeToggle = $('themeToggle');
  const themeLabel = themeToggle.querySelector('.theme-label');
  const root = document.documentElement;
  function syncThemeLabel(){
    // 버튼에는 "눌렀을 때 바뀔 모드"를 적는다
    themeLabel.textContent = root.dataset.theme === 'light' ? '다크 모드' : '화이트 모드';
  }
  syncThemeLabel();
  themeToggle.addEventListener('click', () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = next;
    try{ localStorage.setItem('theme', next); }catch(e){}
    syncThemeLabel();
  });

  function setAdvStatus(){
    advStatus.textContent = apiBaseUrl
      ? ('현재: 실 서버 모드 → ' + apiBaseUrl)
      : '현재: 데모 모드 (서버 주소 미설정)';
  }
  setAdvStatus();
  if (DEFAULT_API_BASE_URL) apiUrlInput.value = DEFAULT_API_BASE_URL;

  saveUrlBtn.addEventListener('click', () => {
    apiBaseUrl = apiUrlInput.value.trim().replace(/\/$/, '');
    setAdvStatus();
  });

  testUrlBtn.addEventListener('click', async () => {
    if (!apiBaseUrl){ advStatus.textContent = '먼저 서버 주소를 입력하고 "이 세션에 적용"을 눌러주세요.'; return; }
    advStatus.textContent = '연결 확인 중…';
    try{
      const res = await fetch(apiBaseUrl, { method:'GET', mode:'cors' });
      advStatus.textContent = '연결 응답 확인됨 (status ' + res.status + ') → ' + apiBaseUrl;
    }catch(err){
      advStatus.textContent = '연결 실패: ' + err.message + ' (CORS 설정 또는 서버 상태를 확인하세요)';
    }
  });

  // ---------- stage status panel ----------
  function renderStageList(activeKey, doneKeys){
    stageListEl.innerHTML = '';
    STAGE_DEFS.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'stage-row';
      if (activeKey === s.key) row.classList.add('active');
      if (doneKeys && doneKeys.has(s.key)) row.classList.add('done');
      row.innerHTML =
        '<span class="stage-code">' + s.code + '</span>' +
        '<span class="stage-dot-row"><span class="stage-dot"></span><span class="stage-name">' + s.name + '</span></span>' +
        '<span class="stage-desc">' + s.desc + '</span>';
      row.addEventListener('click', () => { if (stages.length) showStage(i < stages.length ? i : stages.length-1); });
      stageListEl.appendChild(row);
    });
  }
  renderStageList(null, new Set());

  // ---------- sample gallery (real satellite photos) ----------
  const SAMPLES = [
    { id:'airport-apron', label:'공항 계류장의 일곱 여객기', src:'/img/samples/airport-apron.jpg' },
    { id:'harbor-ships-candidate', label:'분주한 산업 항만과 정박한 다수의 선박', src:'/img/samples/harbor-ships-candidate.png' },
    { id:'helicopter-apron', label:'계류장에 대기 중인 다수의 헬리콥터', src:'/img/samples/helicopter-apron.png' },
  ];

  function loadImage(src){
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  SAMPLES.forEach((s) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'sample-card';
    card.innerHTML = '<img src="' + s.src + '" alt="' + s.label + '" loading="lazy">' +
                      '<div class="sc-label">' + s.label + '</div>';
    card.addEventListener('click', async () => {
      document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      const img = await loadImage(s.src);
      selectAsset({
        kind:'sample',
        sampleId:s.id,
        previewUrl:s.src,
        label:s.label,
        width:img.naturalWidth, height:img.naturalHeight,
      });
    });
    sampleRow.appendChild(card);
  });

  // ---------- upload handling ----------
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); fileInput.click(); } });
  ['dragover','dragenter'].forEach(evt => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
  ['dragleave','dragend'].forEach(evt => dropzone.addEventListener(evt, () => dropzone.classList.remove('drag')));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault(); dropzone.classList.remove('drag');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  function handleFile(file){
    if (!file.type.startsWith('image/')){ runMsg.textContent = '이미지 파일만 업로드할 수 있어요.'; runMsg.classList.add('error'); return; }
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        selectAsset({
          kind:'upload',
          file,
          previewUrl:reader.result,
          label:file.name,
          width:img.naturalWidth, height:img.naturalHeight,
          getBlob: () => Promise.resolve(file),
        });
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

  function selectAsset(asset){
    selected = asset;
    previewThumb.src = asset.previewUrl;
    previewThumb.style.display = 'block';
    previewMeta.textContent = asset.label + ' · ' + asset.width + '×' + asset.height;
    runMsg.textContent = ''; runMsg.classList.remove('error');

    // 업로드/예시 선택 즉시 원본을 뷰어에 보여준다 (실행 전에도 확인 가능)
    stages = [];
    currentStageIdx = 0;
    compareSection.style.display = 'none';
    demoBadge.style.display = 'none';
    renderStageList(null, new Set());
    viewerCode.textContent = STAGE_DEFS[0].code;
    viewerName.textContent = STAGE_DEFS[0].name;
    viewerCaption.textContent = STAGE_DEFS[0].desc;
    viewerBody.innerHTML = '';
    const img = document.createElement('img');
    img.src = asset.previewUrl; img.alt = asset.label;
    viewerBody.appendChild(img);
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  }

  // ---------- pipeline run ----------
  runBtn.addEventListener('click', runPipeline);

  async function runPipeline(){
    if (isProcessing) return;
    if (!selected){ runMsg.textContent = '먼저 이미지를 올리거나 예시를 선택해주세요.'; runMsg.classList.add('error'); return; }

    isProcessing = true;
    runBtn.disabled = true;
    runMsg.classList.remove('error');
    runMsg.textContent = apiBaseUrl ? '서버에 요청 중…' : '데모 모드로 처리 중…';
    compareSection.style.display = 'none';
    stages = [];

    try{
      let result;
      let usedDemo = false;
      if (apiBaseUrl){
        try{
          result = await callRealApi(selected);
        }catch(err){
          usedDemo = true;
          runMsg.textContent = '서버(' + apiBaseUrl + ')에 연결하지 못해 데모 모드로 대신 보여드려요. (' + err.message + ')';
          result = await runDemoSimulation(selected);
        }
      } else {
        usedDemo = true;
        result = await runDemoSimulation(selected);
      }
      const showsMockBadge = usedDemo || !!result.isMock;
      demoBadge.style.display = showsMockBadge ? 'inline-block' : 'none';
      await revealStages(result, usedDemo);
      runMsg.textContent = usedDemo
        ? '데모 모드로 처리를 마쳤어요.'
        : (result.isMock ? '서버 처리를 마쳤어요. (아직 자리표시 모델입니다)' : '처리를 마쳤어요.');
    }catch(err){
      runMsg.textContent = '처리 중 문제가 발생했어요: ' + err.message;
      runMsg.classList.add('error');
    }finally{
      isProcessing = false;
      runBtn.disabled = false;
    }
  }

  async function callRealApi(asset){
    const form = new FormData();
    if (asset.kind === 'sample' && asset.sampleId){
      // 준비된 샘플은 정적으로 미리 계산해둔 결과가 있으므로, 이미지를 다시 업로드해서
      // GPU 큐를 타지 않고 sample_id만 보낸다 (서버가 큐 없이 즉시 응답).
      form.append('sample_id', asset.sampleId);
    } else {
      const blob = await asset.getBlob();
      form.append('image', blob, (asset.file && asset.file.name) || 'upload.png');
    }
    form.append('model_id', 'warm-start-anonymizer');

    const base = apiBaseUrl.replace(/\/$/, '');
    const submitRes = await fetch(base + '/api/jobs', { method:'POST', body: form });
    if (submitRes.status === 503){
      const body = await submitRes.json().catch(() => ({}));
      throw new Error(body.message || '지금 대기열이 가득 찼어요. 잠시 후 다시 시도해주세요.');
    }
    if (!submitRes.ok){
      const body = await submitRes.json().catch(() => ({}));
      throw new Error(body.message || ('작업 제출 실패 (status ' + submitRes.status + ')'));
    }
    const submitData = await submitRes.json();
    const jobId = submitData.job_id;

    let statusData;
    for(;;){
      await sleep(700);
      const statusRes = await fetch(base + '/api/jobs/' + jobId);
      if (!statusRes.ok) throw new Error('상태 조회 실패 (status ' + statusRes.status + ')');
      statusData = await statusRes.json();
      if (statusData.status === 'queued'){
        runMsg.textContent = '대기열에서 처리를 기다리는 중… (대기 순번 ' + (statusData.queue_position ?? '-') + ')';
        continue;
      }
      if (statusData.status === 'processing'){
        runMsg.textContent = '서버에서 처리 중…';
        continue;
      }
      break;
    }

    if (statusData.status === 'error') throw new Error(statusData.error_message || '처리 중 오류가 발생했어요.');
    const data = statusData.result;
    if (!data || !data.result_image) throw new Error('응답 형식이 예상과 달라요 (result_image 없음)');
    return {
      inputUrl: asset.previewUrl,
      boxesUrl: data.boxes_image || asset.previewUrl,
      maskUrl: data.mask_image || null,
      fillUrl: data.coarse_fill_image || null,
      diffuseUrl: data.diffusion_preview_image || data.result_image,
      resultUrl: data.result_image,
      meta: data.meta || {},
      isMock: !!(data.meta && data.meta.mock),
    };
  }

  // ---------- demo (client-side simulated) pipeline ----------
  async function runDemoSimulation(asset){
    const img = await loadImageFromAsset(asset);
    const W = img.naturalWidth || img.width, H = img.naturalHeight || img.height;

    const cx = W * (0.32 + Math.random()*0.28);
    const cy = H * (0.32 + Math.random()*0.28);
    const bw = W * 0.22, bh = H * 0.14;
    const angle = (Math.random()*30 - 15) * Math.PI/180;

    const base = () => { const c = document.createElement('canvas'); c.width=W; c.height=H; return c; };

    // 00 input
    const inputC = base(); inputC.getContext('2d').drawImage(img,0,0,W,H);

    // 01 detect: original + rotated dashed box + label
    const detectC = base(); const dctx = detectC.getContext('2d');
    dctx.drawImage(img,0,0,W,H);
    dctx.save(); dctx.translate(cx,cy); dctx.rotate(angle);
    dctx.strokeStyle = '#E8664F'; dctx.lineWidth = Math.max(2, W*0.006); dctx.setLineDash([8,6]);
    dctx.strokeRect(-bw/2,-bh/2,bw,bh);
    dctx.restore();
    dctx.fillStyle = '#E8664F'; dctx.font = Math.max(12, W*0.02) + 'px IBM Plex Mono, monospace';
    dctx.fillText('object 0.9' + Math.floor(Math.random()*9), cx - bw/2, cy - bh/2 - 6);

    // 02 mask: black bg + white rotated rect (dilated)
    const maskC = base(); const mctx = maskC.getContext('2d');
    mctx.fillStyle = '#000'; mctx.fillRect(0,0,W,H);
    mctx.save(); mctx.translate(cx,cy); mctx.rotate(angle);
    mctx.fillStyle = '#fff'; mctx.fillRect(-bw*0.58,-bh*0.62,bw*1.16,bh*1.24);
    mctx.restore();

    // 03 fill: original with blurred patch inside rotated rect
    const fillC = base(); const fctx = fillC.getContext('2d');
    fctx.drawImage(img,0,0,W,H);
    fctx.save(); fctx.translate(cx,cy); fctx.rotate(angle);
    fctx.beginPath(); fctx.rect(-bw*0.55,-bh*0.6,bw*1.1,bh*1.2); fctx.clip();
    fctx.rotate(-angle); fctx.translate(-cx,-cy);
    fctx.filter = 'blur(' + Math.max(6, W*0.02) + 'px) saturate(0.85)';
    fctx.drawImage(img,0,0,W,H);
    fctx.restore();

    // 04/05 diffuse+output: similar but softer + slight tint (simulated "refinement")
    const outC = base(); const octx = outC.getContext('2d');
    octx.drawImage(img,0,0,W,H);
    octx.save(); octx.translate(cx,cy); octx.rotate(angle);
    octx.beginPath(); octx.rect(-bw*0.55,-bh*0.6,bw*1.1,bh*1.2); octx.clip();
    octx.rotate(-angle); octx.translate(-cx,-cy);
    octx.filter = 'blur(' + Math.max(3, W*0.008) + 'px) saturate(1.03) contrast(1.02)';
    octx.drawImage(img,0,0,W,H);
    octx.restore();

    await sleep(250); // 아주 짧은 지연 — 실제 처리 흐름 느낌만 전달

    return {
      inputUrl: inputC.toDataURL('image/png'),
      boxesUrl: detectC.toDataURL('image/png'),
      maskUrl: maskC.toDataURL('image/png'),
      fillUrl: fillC.toDataURL('image/png'),
      diffuseUrl: outC.toDataURL('image/png'),
      resultUrl: outC.toDataURL('image/png'),
      meta: { detections: 1, nfe: 10, latency_ms: null, demo:true },
    };
  }

  function loadImageFromAsset(asset){
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = asset.previewUrl;
    });
  }

  // ---------- reveal stages sequentially ----------
  async function revealStages(result, isDemo){
    const urlByKey = {
      input: result.inputUrl,
      detect: result.boxesUrl,
      mask: result.maskUrl,
      fill: result.fillUrl,
      diffuse: result.diffuseUrl,
      output: result.resultUrl,
    };
    stages = STAGE_DEFS.map(s => ({ ...s, imgUrl: urlByKey[s.key] }))
                        .filter(s => !!s.imgUrl);

    const done = new Set();
    for (let i = 0; i < stages.length; i++){
      renderStageList(stages[i].key, done);
      showStage(i);
      done.add(stages[i].key);
      await sleep(i === stages.length - 1 ? 0 : 260);
    }
    renderStageList(null, new Set(stages.map(s => s.key)));

    // compare section
    compareBefore.src = result.inputUrl;
    compareAfter.src = result.resultUrl;
    downloadBtn.href = result.resultUrl;
    compareSection.style.display = 'block';
    resetHandle();
  }

  function showStage(idx){
    if (!stages.length) return;
    currentStageIdx = Math.max(0, Math.min(idx, stages.length - 1));
    const s = stages[currentStageIdx];
    viewerCode.textContent = s.code;
    viewerName.textContent = s.name;
    viewerCaption.textContent = s.desc;
    viewerBody.innerHTML = '';
    const img = document.createElement('img');
    img.src = s.imgUrl; img.alt = s.name;
    viewerBody.appendChild(img);
    prevBtn.disabled = currentStageIdx === 0;
    nextBtn.disabled = currentStageIdx === stages.length - 1;
  }
  prevBtn.addEventListener('click', () => showStage(currentStageIdx - 1));
  nextBtn.addEventListener('click', () => showStage(currentStageIdx + 1));

  // ---------- before/after slider ----------
  function setHandle(pct){
    pct = Math.max(0, Math.min(100, pct));
    // 결과(after) 레이어를 핸들 기준 오른쪽에서 보여준다 (왼쪽="원본"/오른쪽="결과" 라벨과 맞춤).
    compareAfterWrap.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
    compareHandle.style.left = pct + '%';
  }
  function resetHandle(){ setHandle(50); }
  let dragging = false;
  compareHandle.addEventListener('pointerdown', (e) => { dragging = true; compareHandle.setPointerCapture(e.pointerId); });
  window.addEventListener('pointerup', () => dragging = false);
  compareSlider.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const rect = compareSlider.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setHandle(pct);
  });

})();
