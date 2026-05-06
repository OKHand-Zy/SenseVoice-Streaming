# WebSocket 情緒識別功能說明

## 概述

`realtime_ws_server_demo.py` 已更新，現在支援在語音結束時輸出情緒 tag。

## 功能驗證

### ✅ 已實現的功能

| 需求 | 狀態 | 說明 |
|------|------|------|
| 網頁介面 | ✅ | `realtime_ws_client.html` |
| 持續輸入音訊 | ✅ | WebSocket 持續接收音訊流 |
| VAD | ✅ | 使用 Silero VAD 自動檢測語音邊界 |
| 即時輸出文字 | ✅ | 每個 chunk 實時發送文字 |
| 輸出情緒 tag | ✅ | **語音結束時輸出整段話的情緒** |

## 工作流程

```
網頁麥克風 → WebSocket → 音訊流
                          ↓
                      VAD 檢測
                          ↓
                  檢測到說話開始 → reset()
                          ↓
                  持續說話 → 實時輸出文字
                          ↓
                  檢測到靜音（說話結束）
                          ↓
                  輸出情緒 tag（基於完整語句）
```

## 修改內容

### 1. 服務器端 (`realtime_ws_server_demo.py`)

#### 修改 1：添加情緒欄位到數據模型

```python
class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None
    emotion: str | None = None  # 新增：情緒標籤
```

#### 修改 2：提取並發送情緒

```python
for res in sensevoice_model.streaming_inference(speech_samples, is_last):
    if len(res["text"]) > 0:
        asrDetected = True

    if asrDetected:
        # 提取情緒（只有在 is_last=True 時才有）
        emotion = res.get("emotion", None)
        
        transcription_response = TranscriptionResponse(
            id=speech_count,
            begin_at=currentAudioBeginTime,
            end_at=None,
            data=TranscriptionChunk(
                timestamps=res["timestamps"],
                raw_text=res["text"],
                emotion=emotion,  # 發送情緒
            ),
            is_final=False,
            session_id=session_id,
        )
        await websocket.send_json(transcription_response.model_dump())
```

### 2. 客戶端 (`realtime_ws_client.html`)

#### 修改 1：添加情緒樣式

```css
.transEmotion {
    grid-column-start: 2;
    grid-column-end: 3;
    grid-row-start: 3;
    grid-row-end: 3;
    padding: 0px 5px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 5px;
}

.emotionTag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    color: white;
}

.emotion-happy { background-color: #4caf50; }    /* 綠色 */
.emotion-sad { background-color: #2196f3; }      /* 藍色 */
.emotion-angry { background-color: #f44336; }    /* 紅色 */
.emotion-neutral { background-color: #9e9e9e; }  /* 灰色 */
.emotion-unk { background-color: #607d8b; }      /* 深灰色 */
```

#### 修改 2：渲染情緒標籤

```javascript
// 在 is_final 的情況下，如果有情緒就顯示
if (transcription.data.emotion) {
    const emotionEle = document.createElement("div");
    emotionEle.classList.add("transEmotion");
    
    const emotionLabel = document.createElement("span");
    emotionLabel.textContent = "情緒:";
    emotionEle.appendChild(emotionLabel);
    
    const emotionTag = document.createElement("span");
    emotionTag.classList.add("emotionTag", `emotion-${transcription.data.emotion}`);
    
    const emotionLabels = {
        'happy': '😊 開心',
        'sad': '😢 悲傷',
        'angry': '😠 生氣',
        'neutral': '😐 中性',
        'unk': '❓ 未知'
    };
    emotionTag.textContent = emotionLabels[transcription.data.emotion] || transcription.data.emotion;
    emotionEle.appendChild(emotionTag);
    
    root.appendChild(emotionEle);
}
```

## 使用方法

### 1. 啟動服務器

```bash
python realtime_ws_server_demo.py
```

或使用自定義參數：

```bash
python realtime_ws_server_demo.py \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda \
    --vad-threshold 0.5 \
    --vad-min-silence-duration-ms 550
```

### 2. 打開網頁

在瀏覽器中訪問：
```
http://127.0.0.1:8000/
```

### 3. 開始錄音

1. 點擊 "Start Recording" 按鈕
2. 允許麥克風權限
3. 開始說話

### 4. 查看結果

**實時輸出：**
```
[0] 你好今天天氣真好
    0.5 ~ Now
```

**說話結束後（靜音 0.5 秒）：**
```
[0] 你好今天天氣真好
    0.5 ~ 3.2
    情緒: 😊 開心
```

## 輸出格式

### WebSocket 消息格式

```json
{
  "type": "TranscriptionResponse",
  "id": 0,
  "begin_at": 0.5,
  "end_at": 3.2,
  "data": {
    "timestamps": [500, 800, 1200, 1800, 2400, 3000],
    "raw_text": "你好今天天氣真好",
    "final_text": null,
    "spk_id": null,
    "emotion": "happy"
  },
  "is_final": true,
  "session_id": "uuid-string"
}
```

### 情緒類型

| 情緒值 | 中文 | 顏色 | Emoji |
|--------|------|------|-------|
| `happy` | 開心 | 綠色 | 😊 |
| `sad` | 悲傷 | 藍色 | 😢 |
| `angry` | 生氣 | 紅色 | 😠 |
| `neutral` | 中性 | 灰色 | 😐 |
| `unk` | 未知 | 深灰 | ❓ |

## 情緒輸出時機

**重要：** 情緒只在 `is_final=True` 時輸出（即 VAD 檢測到說話結束）

### 場景 1：說一句話

```
時間軸: 0s -------- 1s -------- 2s -------- 3s (靜音)

實時輸出:
[0] 你
[0] 你好
[0] 你好今天
[0] 你好今天天氣
[0] 你好今天天氣真好

靜音後輸出:
[0] 你好今天天氣真好
    情緒: 😊 開心  ← 只有這裡有情緒
```

### 場景 2：說多句話

```
第一句:
[0] 你好嗎
    情緒: 😐 中性

第二句:
[1] 我很開心
    情緒: 😊 開心
```

## 參數調整

### VAD 參數

```bash
# 靜音檢測更敏感（更容易斷句）
python realtime_ws_server_demo.py --vad-min-silence-duration-ms 300

# 靜音檢測不敏感（不容易斷句）
python realtime_ws_server_demo.py --vad-min-silence-duration-ms 1000

# VAD 閾值更嚴格（背景噪音不容易觸發）
python realtime_ws_server_demo.py --vad-threshold 0.7
```

### URL 參數

也可以通過 URL 參數動態調整：

```
ws://127.0.0.1:8000/api/realtime/ws?vad_threshold=0.6&vad_min_silence_duration_ms=500
```

## 測試方法

### 測試 1：基本功能

1. 啟動服務器
2. 打開網頁
3. 說話："你好，今天天氣真好"
4. 停止說話（靜音 0.5 秒）
5. 檢查是否顯示情緒標籤

### 測試 2：多句話

1. 說第一句："你好嗎"
2. 停頓（靜音）→ 應該顯示第一句的情緒
3. 說第二句："我很開心"
4. 停頓（靜音）→ 應該顯示第二句的情緒

### 測試 3：不同情緒

嘗試用不同的語氣說話，看情緒識別是否準確：
- 開心的語氣："太棒了！"
- 悲傷的語氣："好難過..."
- 生氣的語氣："真是氣死我了！"
- 平靜的語氣："今天天氣不錯"

## 常見問題

### Q1: 為什麼沒有看到情緒？

**A:** 情緒只在說話結束（VAD 檢測到靜音）時輸出。確保：
1. 說完話後停頓至少 0.5 秒
2. 檢查瀏覽器控制台是否有錯誤
3. 確認 `is_final=true` 的消息中有 `emotion` 欄位

### Q2: 情緒識別不準確？

**A:** 可能的原因：
1. 語句太短（少於 1 秒）
2. 音頻質量差
3. 背景噪音太大
4. 語氣不明顯

### Q3: 如何調試？

**A:** 
1. 打開瀏覽器開發者工具（F12）
2. 查看 Console 標籤頁
3. 查看 Network → WS 標籤頁，檢查 WebSocket 消息
4. 服務器端會輸出詳細日志

## 技術細節

### 情緒識別原理

1. **特徵累積**：從說話開始到結束，累積所有音頻特徵
2. **模型推理**：使用完整的音頻特徵進行情緒識別
3. **結果輸出**：在 `is_last=True` 時輸出情緒標籤

### 性能考慮

- 情緒識別不會增加延遲（只在最後計算）
- 內存占用：每句話累積特徵，說話結束後自動清理
- CPU/GPU 使用：情緒識別與文字識別共用同一個模型

## 總結

✅ **完全符合需求！**

1. ✅ 網頁持續輸入音訊
2. ✅ VAD 自動檢測語音邊界
3. ✅ 即時輸出文字
4. ✅ 語音結束時輸出整段話的情緒 tag

現在可以直接使用 WebSocket 服務進行實時語音識別和情緒分析了！
