"""Generate sample WAV test files."""
import sys, os, struct, math, wave, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def make_wav(path, freq=440, duration=2.0, sr=16000):
    n = int(sr * duration)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        for i in range(n):
            v = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sr))
            wf.writeframes(struct.pack("<h", v))

os.makedirs("data/audio_samples", exist_ok=True)
for name, freq, dur in [
    ("test_order_status.wav", 440, 1.5), ("test_refund.wav", 523, 1.5),
    ("test_support.wav", 659, 1.5), ("test_noisy.wav", 880, 2.0),
    ("test_short.wav", 330, 0.5),
]:
    p = f"data/audio_samples/{name}"
    make_wav(p, freq, dur)
    print(f"Generated: {p}")

with open("data/eval_set.json", "w") as f:
    json.dump([
        {"audio": "data/audio_samples/test_order_status.wav", "reference": "where is my order"},
        {"audio": "data/audio_samples/test_refund.wav", "reference": "i need a refund"},
    ], f, indent=2)
print("Done.")
