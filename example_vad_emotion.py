#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：使用內建 VAD 自動檢測情緒

這個示例展示了兩種使用方式：
1. 啟用內建 VAD（自動檢測說話開始/結束）
2. 手動控制 is_last（向後兼容）
"""

import sounddevice as sd
from streaming_sensevoice import StreamingSenseVoice


def example_with_builtin_vad():
    """示例 1：使用內建 VAD（推薦）"""
    print("=" * 60)
    print("示例 1：使用內建 VAD 自動檢測情緒")
    print("=" * 60)
    
    # 啟用 VAD
    model = StreamingSenseVoice(
        enable_vad=True,
        vad_threshold=0.5,  # VAD 閾值，越高越嚴格
        vad_min_silence_duration_ms=550,  # 靜音多久算說話結束
    )
    
    print("開始錄音，請說話...")
    print("VAD 會自動檢測你何時開始和停止說話")
    print("按 Ctrl+C 停止\n")
    
    samples_per_read = int(0.1 * 16000)  # 每次讀取 0.1 秒
    
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as stream:
        try:
            while True:
                samples, _ = stream.read(samples_per_read)
                audio_int16 = (samples[:, 0] * 32768).astype(int).tolist()
                
                # 不需要指定 is_last，VAD 會自動處理
                for res in model.streaming_inference(audio_int16):
                    if res["text"]:
                        print(f"文字: {res['text']}")
                    
                    # 只有在 VAD 檢測到說話結束時才會有 emotion
                    if "emotion" in res:
                        print(f"✨ 情緒: {res['emotion']}")
                        print("-" * 40)
                        
        except KeyboardInterrupt:
            print("\n錄音結束")


def example_manual_control():
    """示例 2：手動控制（向後兼容）"""
    print("=" * 60)
    print("示例 2：手動控制 is_last")
    print("=" * 60)
    
    # 不啟用 VAD
    model = StreamingSenseVoice(enable_vad=False)
    
    print("開始錄音，請說話...")
    print("每 3 秒會強制結束一次，輸出情緒")
    print("按 Ctrl+C 停止\n")
    
    samples_per_read = int(0.1 * 16000)
    chunk_count = 0
    utterance_duration = 3.0  # 每 3 秒算一句話
    chunks_per_utterance = int(utterance_duration / 0.1)
    
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as stream:
        try:
            while True:
                samples, _ = stream.read(samples_per_read)
                audio_int16 = (samples[:, 0] * 32768).astype(int).tolist()
                
                chunk_count += 1
                is_last = (chunk_count % chunks_per_utterance == 0)
                
                if is_last:
                    print(f"\n[{chunk_count * 0.1:.1f}s] 強制結束，計算情緒...")
                    model.reset()  # 重置以開始新的一句話
                
                for res in model.streaming_inference(audio_int16, is_last=is_last):
                    if res["text"]:
                        print(f"文字: {res['text']}")
                    
                    if "emotion" in res:
                        print(f"✨ 情緒: {res['emotion']}")
                        print("-" * 40)
                        
        except KeyboardInterrupt:
            print("\n錄音結束")


def example_with_audio_file():
    """示例 3：處理音頻文件（使用 VAD）"""
    print("=" * 60)
    print("示例 3：處理音頻文件（使用 VAD）")
    print("=" * 60)
    
    import soundfile as sf
    
    # 啟用 VAD
    model = StreamingSenseVoice(enable_vad=True)
    
    # 讀取音頻文件
    samples, sr = sf.read("data/test_16k.wav")
    samples = (samples * 32768).astype(int).tolist()
    
    print(f"處理音頻文件，長度: {len(samples) / sr:.2f} 秒\n")
    
    # 每次送 0.1 秒的音頻
    step = int(0.1 * sr)
    for i in range(0, len(samples), step):
        audio_chunk = samples[i : i + step]
        
        # VAD 會自動檢測說話片段
        for res in model.streaming_inference(audio_chunk):
            if res["text"]:
                print(f"文字: {res['text']}")
            
            if "emotion" in res:
                print(f"✨ 情緒: {res['emotion']}")
                print("-" * 40)


if __name__ == "__main__":
    import sys
    
    print("\n請選擇示例：")
    print("1. 使用內建 VAD（推薦）")
    print("2. 手動控制 is_last")
    print("3. 處理音頻文件")
    
    choice = input("\n請輸入選項 (1/2/3): ").strip()
    
    if choice == "1":
        example_with_builtin_vad()
    elif choice == "2":
        example_manual_control()
    elif choice == "3":
        example_with_audio_file()
    else:
        print("無效選項")
        sys.exit(1)
