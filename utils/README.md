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
