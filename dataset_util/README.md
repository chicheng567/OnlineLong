# Video Dataset Utilities

## check_missing_videos.py

檢查您的視頻資料集完整性的工具腳本。會掃描標註文件中的所有視頻，並檢查實際的視頻文件是否存在。

### 基本用法

```bash
# 檢查所有資料集
python utils/check_missing_videos.py

# 檢查特定資料集
python utils/check_missing_videos.py --datasets anet_dvc_train youcook_dvc_train

# 顯示詳細進度
python utils/check_missing_videos.py --verbose

# 自動修復：移除缺失視頻的標註條目
python utils/check_missing_videos.py --fix-missing
```

### 參數說明

- `--config`: 資料集配置文件路徑 (預設: `anno_data/finetune_online.json`)
- `--output`: 詳細報告輸出路徑 (預設: `utils/video_check_report.json`)
- `--verbose`: 顯示詳細檢查進度
- `--fix-missing`: 自動從標註文件中移除缺失視頻的條目 (會先備份原文件)
- `--datasets`: 指定要檢查的資料集名稱

### 輸出說明

腳本會輸出：

1. **終端顯示**: 每個資料集的檢查結果和整體統計
2. **JSON報告**: 詳細的檢查報告 (儲存到 `utils/video_check_report.json`)
3. **按資料集分類的缺失視頻清單**: 儲存到 `utils/video_check_report_missing_by_dataset.json`
4. **備份文件**: 如果使用 `--fix-missing`，會創建 `.backup` 檔案

### 範例輸出

```
============================================================
Checking dataset: anet_dvc_train
============================================================
Annotation file: anno_online/dense_video_captioning/anet.json
Data root: /workspace/chicheng/datasets/ActivityNet/videos
Total videos in annotation: 1000

Summary:
  Total videos: 1000
  Existing videos: 950
  Missing videos: 50
  Success rate: 95.0%

Missing videos (showing first 10):
  - v_missing_001.mp4
  - v_missing_002.mp4
  ...
```

### 注意事項

1. 腳本會自動處理無副檔名的文件，嘗試常見的視頻格式
2. 使用 `--fix-missing` 時會自動創建備份文件
3. 支援多種標註文件格式 (list 格式和 dict 格式)
4. 檢查過程中會顯示進度 (每 100 個文件顯示一次)
5. **新功能**: 自動生成按資料集分類的缺失視頻清單

---

## check_video_integrity.py

檢查視頻文件完整性和是否損壞的高效能工具腳本。使用多種方法驗證視頻文件的完整性，包括檔案標頭檢查、FFprobe 分析和 OpenCV 讀取測試。

### 基本用法

```bash
# 檢查所有資料集的視頻完整性（完整模式）
python utils/check_video_integrity.py

# 檢查特定資料集
python utils/check_video_integrity.py --datasets anet_dvc_train youcook_dvc_train

# 快速檢查模式（僅檢查檔案標頭）
python utils/check_video_integrity.py --quick-check

# 隨機取樣檢查（每個資料集檢查 100 個視頻）
python utils/check_video_integrity.py --sample 100

# 使用 8 個進程並行檢查
python utils/check_video_integrity.py --processes 8 --verbose
```

### 參數說明

- `--config`: 資料集配置文件路徑 (預設: `anno_data/finetune_online.json`)
- `--output`: 詳細報告輸出路徑 (預設: `utils/video_integrity_report.json`)
- `--verbose`: 顯示詳細檢查進度
- `--quick-check`: 僅執行快速檔案標頭檢查（速度較快）
- `--sample N`: 每個資料集隨機抽樣 N 個視頻進行檢查
- `--datasets`: 指定要檢查的資料集名稱
- `--processes`: 並行進程數量 (預設: 4)

### 檢查方法

1. **檔案標頭檢查**: 驗證視頻文件的魔數簽名
2. **FFprobe 分析**: 使用 FFprobe 解析視頻元資料
3. **OpenCV 讀取**: 嘗試用 OpenCV 讀取視頻屬性和第一幀

### 輸出說明

腳本會輸出：

1. **終端顯示**: 每個資料集的檢查結果、完整性評分和統計資料
2. **詳細 JSON 報告**: 包含所有檢查結果的完整報告 (儲存到 `utils/video_integrity_report.json`)
3. **按資料集分類的損壞視頻清單**: 簡化版本儲存到 `utils/video_integrity_report_corrupted_by_dataset.json`

### 範例輸出

```
============================================================
Checking dataset: anet_dvc_train
============================================================
Annotation file: anno_online/dense_video_captioning/anet.json
Data root: /workspace/chicheng/datasets/ActivityNet/videos
Check mode: Full (FFprobe + OpenCV + headers)
Processes: 4
Checking 10009 videos...
Starting parallel video integrity checks...

Summary for anet_dvc_train:
  Total videos checked: 10009
  Valid videos: 9950
  Corrupted/damaged videos: 59
  Success rate: 99.4%
  Check duration: 245.2 seconds

Corrupted videos in anet_dvc_train (showing first 10):
  ❌ v_corrupted_001.mp4: FFprobe: No video streams found
  ❌ v_corrupted_002.mp4: OpenCV: Cannot read first frame
  ...
```

### 性能特點

1. **多進程並行**: 使用 multiprocessing 進行高效並行檢查
2. **進度追蹤**: 即時顯示檢查進度
3. **容錯機制**: 處理各種檔案錯誤和異常情況
4. **中斷處理**: 支援 Ctrl+C 中斷，安全結束檢查
5. **資源管理**: 自動釋放視頻資源，避免內存泄漏

### 依賴需求

- **必要**: Python 3.6+, 標準庫
- **可選**: 
  - OpenCV (`pip install opencv-python`) - 用於視頻讀取檢查
  - FFprobe (通常隨 FFmpeg 安裝) - 用於視頻元資料分析

### 注意事項

1. 如果沒有安裝 OpenCV 或 FFprobe，腳本會跳過相應的檢查方法
2. `--quick-check` 模式速度最快，但檢查精度較低
3. 建議根據資料集大小調整 `--processes` 參數
4. 大型資料集建議使用 `--sample` 進行抽樣檢查
5. **新功能**: 自動生成按資料集分類的損壞視頻清單，方便後續處理

---

## vision_patch_analyze.py

Vision Patch 語義分析工具，用於分析 VideoLLaMA3 模型的 vision encoder 輸出的語義特徵。

### 功能概述

#### 1. Frame-to-Frame 語義相似度分析
- **輸出**: `frame_similarity_heatmap.png`
- 對每個 frame 的所有 patches 進行 average pooling
- 計算所有 frame pairs 的 cosine similarity
- 生成熱力圖顯示時間維度的語義相似性
- **用途**: 檢測場景變化、重複內容、視頻結構分析

#### 2. Patch 時間相似度分析 (NEW!)
- **輸出**: `patch_temporal_similarity_analysis.png`
- 分析同一 patch 位置在不同 frames 之間的語義相似度變化
- 對比各個 patch 的時間相似度趨勢與整體 frame 相似度
- **內容**:
  - **頂部**: 時間相似度趨勢圖（Frame-level vs 多個 Patch-level）
  - **中部**: 空間相關性熱力圖（各 patch 與 frame-level 的相關係數）
  - **底部**: 每個 patch 的完整相似度矩陣（網格排列）
- **用途**:
  - 理解不同空間位置的時間穩定性
  - 識別哪些區域對場景變化更敏感
  - 評估 patch-level 和 frame-level 語義的一致性

#### 3. Token-level 語義分析
對每個選定的 frame 生成包含 3 個子圖的分析圖：
- **左圖**: Token-to-Token 相似度矩陣（熱力圖格式，1024×1024）
- **中圖**: PCA 3D 可視化（顯示解釋方差比例）
- **右圖**: 原始視頻幀作為參考

### 快速開始

```bash
# 方法 1: 使用快速腳本（推薦）
./run_vision_analysis.sh

# 方法 2: 指定視頻和輸出目錄
./run_vision_analysis.sh path/to/video.mp4 output_directory

# 方法 3: 使用環境變量控制參數
FPS=2 MAX_FRAMES=50 NUM_SAMPLE_FRAMES=8 NUM_PATCHES_ANALYZE=25 ./run_vision_analysis.sh video.mp4

# 方法 4: 直接使用 Python
export PYTHONPATH=.
/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --video_path v__7a80bvsbk8.mp4 \
    --output_dir vision_patch_analysis \
    --max_frames 100 \
    --num_sample_frames 8
```

### 參數說明

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--video_path` | `v__7a80bvsbk8.mp4` | 輸入視頻文件路徑 |
| `--model_path` | `pretrained_models/videollama3_7b_local` | 預訓練模型路徑 |
| `--output_dir` | `vision_patch_analysis` | 輸出目錄 |
| `--fps` | `1` | 視頻採樣率（每秒幀數） |
| `--max_frames` | `200` | 最大處理幀數 |
| `--device` | `cuda:0` | 使用的設備 |
| `--num_sample_frames` | `5` | 均勻採樣分析的幀數 |
| `--sample_frames` | `None` | 指定特定幀索引（例: `0 10 20`） |
| `--num_patches_analyze` | `16` | Patch 時間分析的 patch 數量（例: 16 為 4×4 網格） |

### 輸出文件

```
vision_patch_analysis/
├── frame_similarity_heatmap.png              # Frame-to-frame 相似度矩陣
├── patch_temporal_similarity_analysis.png    # Patch 時間相似度分析 (NEW!)
├── frame_0000_token_analysis.png             # Frame 0 的 token 分析
├── frame_0012_token_analysis.png             # Frame 12 的 token 分析
└── ...
```

### 使用範例

```bash
# 範例 1: 快速測試（10 幀，3 個樣本）
/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --video_path v__7a80bvsbk8.mp4 \
    --output_dir quick_test \
    --max_frames 10 \
    --num_sample_frames 3

# 範例 2: 分析特定幀
/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --video_path video.mp4 \
    --sample_frames 0 10 20 30 40

# 範例 3: 高時間解析度（2 FPS，100 幀）
/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --fps 2 \
    --max_frames 100 \
    --num_sample_frames 10

# 範例 4: 分析更多 patch 位置（9×9 = 81 個 patches）
/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --video_path video.mp4 \
    --num_patches_analyze 81 \
    --max_frames 50

# 範例 5: 批量處理多個視頻
for video in *.mp4; do
    /miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
        --video_path "$video" \
        --output_dir "analysis_${video%.mp4}" \
        --max_frames 50
done
```

### 性能建議

**記憶體使用**:
- 10 frames, 3 samples: ~2GB GPU
- 50 frames, 8 samples: ~4GB GPU
- 100 frames, 10 samples: ~6GB GPU

**執行時間**:
- 10 frames: ~30 秒
- 50 frames: ~90 秒
- 100 frames: ~3 分鐘

### 技術細節

- **Cosine Similarity**: `similarity = (A · B) / (||A|| × ||B||)`，範圍 [0, 1]
- **PCA 降維**: 降維到 3 個主成分用於 3D 可視化
- **Vision Token**: 默認 448×448 圖像 → 32×32 = 1024 patches，每個 patch 1152-dim
- **圖像格式**: 自動處理 PIL Image、PyTorch Tensor、NumPy Array

### 使用場景

1. **視頻內容分析**: 檢測場景切換、找出重複內容
2. **模型行為研究**: 理解 vision encoder 的語義表徵
   - **NEW**: 分析不同空間位置的時間穩定性
   - **NEW**: 評估 patch-level 和 frame-level 語義一致性
3. **數據集質量評估**: 檢查視頻多樣性、識別異常幀
4. **Ablation Study**: 比較不同模型配置、評估 token compression 影響
   - **NEW**: 研究不同 patch 位置對時間變化的敏感度

### 依賴需求

- `torch` - PyTorch 深度學習框架
- `numpy` - 數值計算
- `matplotlib` - 繪圖庫
- `seaborn` - 統計數據可視化
- `scikit-learn` - PCA 降維
- VideoLLaMA3（本專案）
