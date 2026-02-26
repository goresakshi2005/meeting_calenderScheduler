"""
Test script for faster-whisper
"""

import sys
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def main():
    print("\n" + "=" * 60)
    print("FASTER-WHISPER TEST")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python test_faster.py <audio_file>")
        print("\nExample: python test_faster.py meeting.mp3")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"\n❌ File not found: {file_path}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"\n📁 Processing: {file_path}")
    print(f"📊 File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n🔄 Loading faster-whisper model...")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ Model loaded")
        
        print("\n🔄 Transcribing...")
        segments, info = model.transcribe(str(file_path.absolute()))
        
        print(f"✅ Transcription complete!")
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        print("\n" + "=" * 60)
        print("TRANSCRIPT:")
        print("=" * 60)
        
        full_text = []
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text.append(segment.text)
        
        transcript = " ".join(full_text)
        output_file = file_path.parent / f"{file_path.stem}_transcript.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        print(f"\n💾 Full transcript saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()