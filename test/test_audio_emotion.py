#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試音檔的情緒識別
"""

import sys
import os

# 將父目錄加入 Python 路徑，以便導入 streaming_sensevoice
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
from streaming_sensevoice import StreamingSenseVoice


def test_audio_file(audio_path):
    """測試單個音檔的情緒識別"""
    print("=" * 70)
    print(f"測試音檔: {audio_path}")
    print("=" * 70)
    
    model = StreamingSenseVoice(enable_vad=True)
    
    try:
        samples, sr = sf.read(audio_path)
        if len(samples.shape) > 1:
            samples = samples[:, 0]  # 取第一個聲道
        samples = (samples * 32768).astype(int).tolist()
        
        print(f"📁 音檔長度: {len(samples) / sr:.2f} 秒")
        print(f"📁 採樣率: {sr} Hz")
        print()
        print("開始處理...")
        print("-" * 70)
        
        step = int(0.1 * sr)
        utterance_count = 0
        current_text_buffer = []
        
        total_chunks = (len(samples) + step - 1) // step
        
        for i in range(0, len(samples), step):
            audio_chunk = samples[i : i + step]
            chunk_idx = i // step
            is_last_chunk = (chunk_idx == total_chunks - 1)
            
            # 如果是最後一個 chunk，添加靜音來觸發 VAD 結束
            if is_last_chunk:
                # 添加 1 秒的靜音
                silence = [0] * sr
                audio_chunk = audio_chunk + silence
            
            for res in model.streaming_inference(audio_chunk):
                if res["text"]:
                    print(res["text"], end="", flush=True)
                    current_text_buffer.append(res["text"])
                
                if "emotion" in res:
                    utterance_count += 1
                    print()
                    print()
                    print(f"📝 完整文字: {''.join(current_text_buffer)}")
                    print(f"✨ 情緒標籤: {res['emotion']}")
                    print(f"📊 句子編號: 第 {utterance_count} 句")
                    print("=" * 70)
                    print()
                    
                    current_text_buffer = []
        
        if utterance_count == 0:
            print("\n⚠️  沒有檢測到任何語音片段")
        else:
            print(f"\n✅ 處理完成，總共識別了 {utterance_count} 句話")
        
    except FileNotFoundError:
        print(f"❌ 找不到音檔: {audio_path}")
    except Exception as e:
        print(f"❌ 處理音檔時發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # 測試所有音檔
    audio_files = [
        "data/angry_16k.wav",
    ]
    
    if len(sys.argv) > 1:
        # 如果提供了命令行參數，測試指定的音檔
        audio_files = sys.argv[1:]
    
    for audio_file in audio_files:
        test_audio_file(audio_file)
        print("\n" * 2)