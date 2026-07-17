# 实施计划：局域网人脸路径 — 被动活体检测（passive single-frame anti-spoofing）

> 交付给执行 agent。主线程（协调方）负责最终**验收 + warehouse 侧接入**。
> 每阶段末尾有**验收标准**，按它产出 EVIDENCE。

## 0. 背景与目标

warehouse 人脸有两条路径：
- **本机模式**：设备(SenseCAP Watcher / Himax WE2)本地识别，上限 **20 人**（NVS 16KB 限制），MobileFaceNet 128D。
- **局域网模式**：设备拍一张 JPEG，POST 到 `face_rec_api`（RPi + Hailo-8 / 未来 Jetson）。大模型(buffalo_l: scrfd_10g + arcface)、向量库无人数上限、更准。

**本计划只做一件净新增的事：给 `face_rec_api` 加被动活体检测**，让局域网路径能挡打印照/屏幕翻拍。识别本身已存在，不用重做。

**约束（来自产品）**：
- 摄像头就一个 RGB，**不推视频流**（带宽/SRAM），**按单帧**推。
- **被动**活体：用户无需眨眼/转头，单张 RGB 判真假。
- **可扩展**：先 Hailo(RPi)，架构要能无痛扩到 Jetson(TensorRT 原生) 和 RKNN。

## 1. 现有架构（已核实，动手前先读源码确认）

仓库 `/Users/harvest/project/face_rec_api`：
- `src/backends/base.py` — `FaceBackend` ABC：`load(detector_path, embedder_path)` / `detect_raw` / `embed_raw` / `detector_input_hw` / `model_tag` / `backend_name` / `close` / `health_check`。
- `src/backends/hailo.py` — `_HailoModelRunner`（单模型异步 runner，共享 VDevice，`vdevice.create_infer_model(hef)` 加载）。加活体 = 再起一个 runner 跑活体 HEF。
- `src/backends/tensorrt.py`（Jetson）、`src/backends/rknn.py`。
- `src/face_pipeline.py` — `FacePipeline`：BGR → `detect_raw` → SCRFD decode+NMS → 5点对齐+112×112 crop → `embed_raw` → 512D。`process_image_base64()` 是入口。
- `src/app.py` — FastAPI：`/infer`、`/recognize`、`/enroll`、`/detect_and_embed`、`/health`、`/list`、`/remove`、`/reload`。`RecognizeResponse{matched,name,confidence}`。
- `src/config.py` — `FACE_BACKEND`(hailo|jetson|rknn)、模型路径 `models/<backend>/{scrfd_10g,arcface_mobilefacenet}.{hef|engine|rknn}`、`SIMILARITY_THRESHOLD=0.4` 等，全走 env。
- `models/hailo/` 已有 `scrfd_10g.hef` + `arcface_mobilefacenet.hef`。`models/jetson`、`models/rknn` 空。
- `tools/` — `build_engine.sh`(jetson)、`build_rknn.py`、`download_insightface.sh`。**无 Hailo 编译脚本**（现有 HEF 是外部编好的）。

## 2. 活体模型选型

**Silent-Face-Anti-Spoofing（MiniFASNet，Minivision 开源，Apache-2.0）**——被动单帧 RGB 的事实标准：
- 输入：**检测框按 scale 外扩后 resize 到 80×80**（注意：**不是**对齐后的 112×112 crop，是原始 bbox 加边距区域；scale 通常 2.7 或 4.0）。
- 输出：softmax，`[fake, real]`（2 类）或 `[2D-spoof, real, 3D-spoof]`（3 类）。边缘端**单个 MiniFASNet(2.7_80x80) 足够**；要更稳可上 2.7 + 4.0 双模型集成（本计划 P0 先单模型，接口预留集成）。
- 体积 ~1-2MB，单帧 CNN，Hailo/TensorRT/RKNN 都能编。

来源：Minivision `Silent-Face-Anti-Spoofing` 仓库的 `.pth` → 导出 ONNX（仓库自带导出脚本或手写）。**执行前先确认能拿到 ONNX 或预编译 HEF**；拿不到就在 P0.1 里解决（导出 + 编译），这是关键前置。

## 3. Phase P0 — face_rec_api 活体 pipeline（RPi + Hailo）

### P0.1 拿到并编译活体模型
- 从 Minivision Silent-Face 仓库取 MiniFASNet 权重 → 导出 ONNX（输入 1×3×80×80）。
- Hailo DFC 编译：ONNX → parse → optimize(量化，用一批真实/假体人脸 crop 做 calib) → compile → `models/hailo/liveness_minifasnet.hef`。
- 在 `tools/` 加编译脚本/文档（仿 `build_engine.sh` / `build_int8_calib.md` 风格），记录 ONNX 来源、输入尺寸、量化 calib 数据、Hailo DFC 版本。
- **卡点**：若本机无 Hailo DFC 环境，在报告里写明，改在 fleet RPi 或有 DFC 的机器上编；不要硬造 HEF。

### P0.2 扩展 backend 抽象
- `base.py`：`FaceBackend` 加抽象方法 `liveness_raw(face_crop_bgr: np.ndarray) -> float`（返回 real 概率 0-1）。`load` 加可选参数 `liveness_path: Optional[str] = None`（None → 该 backend 无活体）。
- `hailo.py`：`load` 里若 `liveness_path` 非空，起第三个 `_HailoModelRunner` 跑活体 HEF。`liveness_raw` 做预处理（bbox 外扩 → 80×80 → 归一化，严格对齐训练时的预处理，务必核对 Minivision 的 transform）→ 推理 → softmax → 返回 real 概率。
- `tensorrt.py` / `rknn.py`：`liveness_raw` 先 `raise NotImplementedError`（P2 填），`load` 忽略 `liveness_path`。

### P0.3 pipeline 插步
- `face_pipeline.py`：detect + 拿到每张脸的**原始 bbox**后（对齐/embed 之前），若 backend 支持活体且 `LIVENESS_ENABLED`，对 bbox 外扩区域调 `backend.liveness_raw` → 得 `liveness_score`。
- 决策：`live = liveness_score >= LIVENESS_THRESHOLD`。`LIVENESS_FAIL_ACTION`：
  - `reject`（默认）：假体的脸不进入 embed/识别，视为未识别。
  - `flag`：照常识别但把 `live=false` 带回，交上层决定。
- 把 `liveness_score` + `live` 挂到每张脸的结果里。**注意 bbox 外扩要 clamp 到图像边界**。

### P0.4 config
`config.py` 加：
- `FACE_LIVENESS_MODEL` = `models/<backend>/liveness_minifasnet.{ext}`（env 可覆盖）。
- `LIVENESS_ENABLED`（bool，默认 true；模型缺失时自动降级为 false 并 warn，不崩）。
- `LIVENESS_THRESHOLD`（float，默认先 0.5，P0 验收后按实测分布调）。
- `LIVENESS_FAIL_ACTION`（`reject`|`flag`，默认 `reject`）。

### P0.5 API
- `RecognizeResponse` / `InferResponse.FaceResult` / `DetectAndEmbedResponse` 加 `live: Optional[bool]` + `liveness_score: Optional[float]`（可选，向后兼容：活体关闭时为 null）。
- `/recognize`：`reject` 模式下假体 → `matched=false`，加一个原因字段（如 `reason:"spoof"` 或复用现有失败语义），别把假体当匹配返回。
- `/health`：`backend`/model 信息里加活体模型状态（loaded / disabled / missing）。

### P0.6 ✅ P0 验收标准
在 fleet RPi(Hailo) 上跑起 face_rec_api（`FACE_BACKEND=hailo`，活体开），用真实场景测：
- **真人正脸** → `live=true`、`liveness_score` 高、正常识别。
- **打印照片**(A4 彩打/手机相册照) → `live=false`、被 reject（`matched=false`）。
- **手机/屏幕翻拍** → `live=false`、被 reject。
EVIDENCE 必含：编译产物 md5、多组样本(真人/打印/屏幕各 ≥5)的 `liveness_score` 原始数值分布、误拒/误收计数、单帧活体推理延迟(ms)、`/health` 输出、`SIMILARITY`/活体阈值配置。据分布给 `LIVENESS_THRESHOLD` 推荐值。

## 4. Phase P1 — warehouse 局域网模式接入（主线程做，执行 agent 不用碰 warehouse）

> 这块我(协调方)接。执行 agent P0 完成即可，P1 我来。留档说明接口即可：
- warehouse 局域网模式已经 POST 到 `<endpoint>/recognize`（设备 RemoteRecognize 或后端 interface）。face_rec_api 在 `reject` 模式下把假体当 `matched=false` 返回，warehouse 现有链路自然 deny。
- 我会：把 tenant_face_config(lan) 的 endpoint 指向 RPi 的 face_rec_api；决定是否把活体判定进 warehouse 审计(`face_auth_logs` 加原因 `spoof`)；在 UI「基础配置」里暴露活体开关/阈值（若需下发）。
- **P1 验收**：warehouse 局域网出库，真人 → 放行；打印照 → 拒(spoof)。

## 5. Phase P2 — 扩展性（Jetson TensorRT，可后置）

- MiniFASNet → TensorRT engine（仿 `tools/build_engine.sh`），实现 `tensorrt.py` 的 `liveness_raw`。RKNN 同理可选。
- **P2 验收**：`FACE_BACKEND=jetson` 活体行为与 Hailo 一致（同批样本 live/spoof 判定一致）。证明"换后端不改 pipeline/业务"。

## 6. 执行护栏（所有阶段）
- **不破坏现有识别**：活体是可选步骤(`LIVENESS_ENABLED`)，关掉时行为与现在完全一致；API 字段向后兼容。
- 活体逻辑**只走 `FaceBackend` 抽象**，不在 pipeline 里写死 Hailo 细节——否则 Jetson 扩展会返工。
- 预处理**必须严格对齐 Minivision 训练时的 transform**（bbox scale、resize、归一化），否则分数漂移、活体形同虚设。这是最容易错的点，P0.6 用真实样本验证。
- 模型缺失/DFC 环境缺失 → 明确报告，不硬造、不静默跳过。
- 每阶段报告附**原始工具输出**（推理分数、延迟、md5），不要只给摘要。
- 禁止改 warehouse_system 仓（P1 主线程做）；禁止 git push / rm -rf / sudo。

## 7. 主线程（我）接入清单（P0 完成后）
1. 验收 P0 的 EVIDENCE（活体分数分布、误拒误收、延迟），定 `LIVENESS_THRESHOLD`。
2. 部署 face_rec_api 到 fleet RPi(Hailo)，健康检查。
3. warehouse 局域网 endpoint 指过去；决定活体是否进审计 + UI 开关。
4. 端到端验收（真人放行/假体拒）。
5. 排 P2(Jetson) 优先级。
