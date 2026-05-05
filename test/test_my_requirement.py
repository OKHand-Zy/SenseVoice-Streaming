#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試：驗證是否符合需求

需求：
1. 持續輸入語音（stream）
2. 持續輸出文字
3. 結束語音時（或靜音時）輸出整段話的情緒 tag
"""

import sounddevice as sd
from streaming_sensevoice import StreamingSenseVoice


def test_requirement():
    """
    測試需求：
    - 持續輸入語音 ✓
    - 持續輸出文字 ✓
    - 靜音時輸出情緒 ✓
    """
    print("=" * 70)
    print("測試：持續語音輸入 + 實時文字輸出 + 靜音時輸出情緒")
    print("=" * 70)
    print()
    print("📝 使用說明：")
    print("1. 對著麥克風說話")
    print("2. 你會看到文字實時輸出")
    print("3. 停止說話（靜音 0.5 秒以上）")
    print("4. 會輸出整段話的情緒 tag")
    print()
    print("按 Ctrl+C 停止程式")
    print("=" * 70)
    print()
    
    # 啟用 VAD，自動檢測靜音
    model = StreamingSenseVoice(
        enable_vad=True,
        vad_threshold=0.5,
        vad_min_silence_duration_ms=500,  # 靜音 0.5 秒算結束
    )
    
    samples_per_read = int(0.1 * 16000)  # 每次讀取 0.1 秒
    utterance_count = 0
    
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as stream:
        try:
            while True:
                # 1. 持續輸入語音（stream）
                samples, _ = stream.read(samples_per_read)
                audio_int16 = (samples[:, 0] * 32768).astype(int).tolist()
                
                for res in model.streaming_inference(audio_int16):
                    # 2. 持續輸出文字
                    if res["text"]:
                        print(f"📝 文字: {res['text']}", end="", flush=True)
                    
                    # 3. 靜音時輸出情緒 tag
                    if "emotion" in res:
                        utterance_count += 1
                        print()  # 換行
                        print(f"✨ 情緒: {res['emotion']}")
                        print(f"📊 這是第 {utterance_count} 句話")
                        print("-" * 70)
                        print()
                        
        except KeyboardInterrupt:
            print("\n\n程式結束")
            print(f"總共識別了 {utterance_count} 句話")


def test_with_visual_feedback():
    """
    測試需求（帶視覺反饋）
    """
    print("=" * 70)
    print("測試：持續語音輸入 + 實時文字輸出 + 靜音時輸出情緒（視覺版）")
    print("=" * 70)
    print()
    print("📝 使用說明：")
    print("1. 對著麥克風說話 → 看到 [🎤 說話中...]")
    print("2. 文字會實時顯示")
    print("3. 停止說話 → 看到 [🔇 靜音中...]")
    print("4. 輸出情緒 tag")
    print()
    print("按 Ctrl+C 停止程式")
    print("=" * 70)
    print()
    
    model = StreamingSenseVoice(
        enable_vad=True,
        vad_threshold=0.5,
        vad_min_silence_duration_ms=500,
    )
    
    samples_per_read = int(0.1 * 16000)
    utterance_count = 0
    current_text_buffer = []
    is_speaking = False
    
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as stream:
        try:
            while True:
                samples, _ = stream.read(samples_per_read)
                audio_int16 = (samples[:, 0] * 32768).astype(int).tolist()
                
                for res in model.streaming_inference(audio_int16):
                    if res["text"]:
                        if not is_speaking:
                            print("\n[🎤 說話中...]", end=" ")
                            is_speaking = True
                        
                        print(res["text"], end="", flush=True)
                        current_text_buffer.append(res["text"])
                    
                    if "emotion" in res:
                        utterance_count += 1
                        is_speaking = False
                        
                        print()
                        print("[🔇 靜音檢測到，輸出情緒...]")
                        print()
                        print(f"📝 完整文字: {''.join(current_text_buffer)}")
                        print(f"✨ 情緒標籤: {res['emotion']}")
                        print(f"📊 句子編號: 第 {utterance_count} 句")
                        print("=" * 70)
                        
                        current_text_buffer = []
                        
        except KeyboardInterrupt:
            print("\n\n程式結束")
            print(f"總共識別了 {utterance_count} 句話")


def test_with_file():
    """
    測試需求（使用音頻文件）
    """
    import soundfile as sf
    
    print("=" * 70)
    print("測試：使用音頻文件驗證功能")
    print("=" * 70)
    print()
    
    model = StreamingSenseVoice(enable_vad=True)
    
    try:
        samples, sr = sf.read("data/test_16k.wav")
        samples = (samples * 32768).astype(int).tolist()
        
        print(f"📁 音頻文件長度: {len(samples) / sr:.2f} 秒")
        print()
        print("開始處理...")
        print("-" * 70)
        print()
        
        step = int(0.1 * sr)
        utterance_count = 0
        current_text_buffer = []
        
        for i in range(0, len(samples), step):
            audio_chunk = samples[i : i + step]
            
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
        
        print(f"\n處理完成，總共識別了 {utterance_count} 句話")
        
    except FileNotFoundError:
        print("❌ 找不到音頻文件 data/test_16k.wav")
        print("請使用麥克風測試模式")


if __name__ == "__main__":
    import sys
    
    print("\n請選擇測試模式：")
    print("1. 基本模式（麥克風）")
    print("2. 視覺反饋模式（麥克風，推薦）")
    print("3. 音頻文件模式")
    
    choice = input("\n請輸入選項 (1/2/3): ").strip()
    print()
    
    if choice == "1":
        test_requirement()
    elif choice == "2":
        test_with_visual_feedback()
    elif choice == "3":
        test_with_file()
    else:
        print("無效選項")
        sys.exit(1)
