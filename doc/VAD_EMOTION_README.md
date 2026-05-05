# VAD 與情緒識別整合說明

## 概述

現在 `StreamingSenseVoice` 支援兩種模式：

1. **內建 VAD 模式**（推薦）：自動檢測說話開始/結束，自動輸出情緒
2. **手動控制模式**：向後兼容，需要手動指定 `is_last` 參數

## 快速開始

### 方式 1：使用內建 VAD（最簡單）

```python
from streaming_sensevoice import StreamingSenseVoice

# 啟用 VAD
model = StreamingSenseVoice(
    enable_vad=True,
    vad_threshold=0.5,
    vad_min_silence_duration_ms=550
)

# 持續送入音頻，不需要指定 is_last
for res in model.streaming_inference(audio_chunk):
    print(res["text"])
    if "emotion" in res:  # VAD 檢測到說話結束時才有
        print(f"情緒: {res['emotion']}")
```

### 方式 2：手動控制（向後兼容）

```python
from streaming_sensevoice import StreamingSenseVoice

# 不啟用 VAD
model = StreamingSenseVoice(enable_vad=False)

# 需要手動指定 is_last
for res in model.streaming_inference(audio_chunk, is_last=True):
    print(res["text"])
    if "emotion" in res:
        print(f"情緒: {res['emotion']}")
```

## VAD 工作原理

### 說話檢測流程

```
音頻流 → VAD 檢測
         ↓
    檢測到說話開始
         ↓
    自動調用 reset()
         ↓
    開始累積音頻特徵
         ↓
    持續輸出文字識別結果
         ↓
    檢測到靜音（說話結束）
         ↓
    自動設置 is_last=True
         ↓
    輸出情緒 tag
```

### 情緒輸出時機

**重要：情緒只在 VAD 檢測到「說話結束」時輸出一次**

- ✅ 基於完整語句的所有音頻特徵
- ✅ 自動檢測說話邊界
- ✅ 不需要手動管理 `reset()` 和 `is_last`

### 示例場景

#### 場景 1：持續說話 5 秒

```
時間軸: 0s -------- 1s -------- 2s -------- 3s -------- 4s -------- 5s
        |                                                            |
    VAD 檢測到                                                  VAD 檢測到
    說話開始                                                    說話結束
    reset()                                                     is_last=True

輸出:
0.0-0.5s: {"text": "你好"}
0.5-1.0s: {"text": "今天"}
1.0-1.5s: {"text": "天氣"}
...
4.5-5.0s: {"text": "真好", "emotion": "happy"}  ← 只有這裡有情緒！
```

#### 場景 2：說兩句話（中間停頓）

```
時間軸: 0s --- 2s --- 3s --- 5s
        |      |      |      |
       說話   停頓   說話   停頓
       
第一句話:
0-2s: "你好嗎" → 2s 時輸出 emotion: "neutral"

第二句話:
3-5s: "我很好" → 5s 時輸出 emotion: "happy"
```

## 參數說明

### `enable_vad` (bool, default: False)
- `True`: 啟用內建 VAD，自動檢測說話邊界
- `False`: 手動控制模式，需要指定 `is_last`

### `vad_threshold` (float, default: 0.5)
- 範圍：0.0 - 1.0
- 越高越嚴格（更不容易觸發）
- 建議值：0.3 - 0.7

### `vad_min_silence_duration_ms` (int, default: 550)
- 靜音多久（毫秒）才算說話結束
- 越大越不容易斷句
- 建議值：300 - 1000

## 完整示例

### 實時麥克風輸入

```python
import sounddevice as sd
from streaming_sensevoice import StreamingSenseVoice

model = StreamingSenseVoice(enable_vad=True)

samples_per_read = int(0.1 * 16000)
with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as stream:
    while True:
        samples, _ = stream.read(samples_per_read)
        audio_int16 = (samples[:, 0] * 32768).astype(int).tolist()
        
        for res in model.streaming_inference(audio_int16):
            if res["text"]:
                print(f"文字: {res['text']}")
            if "emotion" in res:
                print(f"✨ 情緒: {res['emotion']}")
```

### 處理音頻文件

```python
import soundfile as sf
from streaming_sensevoice import StreamingSenseVoice

model = StreamingSenseVoice(enable_vad=True)

samples, sr = sf.read("audio.wav")
samples = (samples * 32768).astype(int).tolist()

step = int(0.1 * sr)
for i in range(0, len(samples), step):
    audio_chunk = samples[i : i + step]
    
    for res in model.streaming_inference(audio_chunk):
        if res["text"]:
            print(f"文字: {res['text']}")
        if "emotion" in res:
            print(f"✨ 情緒: {res['emotion']}")
```

## 情緒標籤

支援的情緒類型：

- `happy`: 開心
- `sad`: 悲傷
- `angry`: 生氣
- `neutral`: 中性
- `unk`: 未知

## 常見問題

### Q1: 為什麼我一直說話但沒有看到情緒？

**A:** 情緒只在 VAD 檢測到「說話結束」時輸出。如果你持續說話不停頓，VAD 不會觸發 `is_last=True`，就不會輸出情緒。

**解決方法：**
- 說完一句話後停頓 0.5-1 秒
- 調整 `vad_min_silence_duration_ms` 參數

### Q2: 情緒識別不準確怎麼辦？

**A:** 情緒識別基於完整語句的音頻特徵。可能的原因：

1. 語句太短（少於 1 秒）
2. 音頻質量差
3. 背景噪音太大

**建議：**
- 確保每句話至少 1-2 秒
- 使用清晰的音頻輸入
- 調整 VAD 參數以獲得更好的斷句

### Q3: 可以不用 VAD，每隔固定時間輸出情緒嗎？

**A:** 可以，使用手動控制模式：

```python
model = StreamingSenseVoice(enable_vad=False)

chunk_count = 0
for audio_chunk in audio_stream:
    chunk_count += 1
    is_last = (chunk_count % 30 == 0)  # 每 3 秒
    
    if is_last:
        model.reset()  # 重置以開始新的片段
    
    for res in model.streaming_inference(audio_chunk, is_last=is_last):
        if "emotion" in res:
            print(f"情緒: {res['emotion']}")
```

### Q4: VAD 和手動控制可以混用嗎？

**A:** 可以，但不建議。如果啟用 VAD 但仍然指定 `is_last`，手動指定的 `is_last` 會優先。

## 運行示例

```bash
# 運行交互式示例
python example_vad_emotion.py

# 選項 1: 使用內建 VAD（推薦）
# 選項 2: 手動控制 is_last
# 選項 3: 處理音頻文件
```

## 依賴

確保已安裝 `pysilero`：

```bash
pip install pysilero
```

或使用完整依賴：

```bash
pip install -r requirements.txt
```
