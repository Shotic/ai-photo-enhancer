# ✨ AI Photo Enhancer (Wink Alternative)

A lightweight, open-source AI image enhancer and upscaler built for both desktop and mobile web (iOS/Android). Reconstructs high-frequency details, sharpens blurry edges, removes noise and compression artifacts, and applies portrait skin softening using deep learning super-resolution.

---

## 📁 Project Architecture

```
ai-photo-enhancer/
├── README.md               # Architecture and setup guide
├── requirements.txt        # Production dependencies (Gradio, ONNX Runtime, Pillow, NumPy)
├── enhancer.py             # Inference engine (Real-ESRGAN v3 ONNX, seamless tiling, EXIF orientation fix)
├── app.py                  # Gradio 6 mobile-optimized web application (with share=True)
└── test_enhancer.py        # Automated test suite (6 test cases covering edge cases & clamping)
```

---

## ⚙️ How It Works (Technical Overview)

1. **Super-Resolution Engine (`enhancer.py`):**
   * Uses an ONNX-optimized export of **Real-ESRGAN General v3** (`realesr-general-x4v3.onnx`).
   * Automatically downloads weights (~4.6 MB) from Hugging Face Hub on first boot and caches them locally.
   * Runs natively via `onnxruntime` across CPU, DirectML (Windows AMD/Intel/Nvidia GPUs), or CUDA without requiring a multi-gigabyte PyTorch installation.

2. **Seamless Tiled Inference:**
   * High-resolution camera photos (e.g. 12MP/48MP from iOS) would normally exhaust RAM if processed in a single pass ($O(H \times W)$ internal CNN activations).
   * The enhancer segments images into $384 \times 384$ overlapping tiles with linear ramp blending at the seams.
   * Space complexity is capped at $O(S^2)$ regardless of input resolution.

3. **iOS Mobile Optimizations:**
   * **EXIF Transposition:** Automatically corrects iOS camera rotation metadata (`ImageOps.exif_transpose`) so uploaded photos never display sideways or inverted.
   * **Touch UI:** 48px minimum touch targets adhering to Apple Human Interface Guidelines.
   * **Save to Photos:** Outputs PNG format; long-press or tap download to save directly into the iOS Camera Roll.

---

## 🚀 Running Locally (Windows / macOS / Linux)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python test_enhancer.py
```

### 3. Launch the Web Server
```bash
python app.py
```

---

## 🌐 Free Remote Access on iOS (Without Hugging Face PRO)

Hugging Face recently restricted creating **new** backend compute Spaces (Gradio and Docker) to paid PRO subscribers ($9/month). 

To use this on your iPhone 100% free without paying:

### Method 1: Instant Public Link (`share=True` on your PC)
In `app.py`, `share=True` is enabled by default. When you run `python app.py`, Gradio automatically generates a secure, encrypted public HTTPS link:
```text
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
```
1. Open the generated `https://xxxx.gradio.live` link directly in **Safari on your iPhone** (works anywhere over cellular or Wi-Fi).
2. Tap **Share → Add to Home Screen** in Safari to save it as an app icon.
3. No accounts, credit cards, or configurations required.

### Method 2: Free Cloud GPU via Google Colab
If you don't want to keep your PC powered on:
1. Open [Google Colab](https://colab.research.google.com/) (Free).
2. Set Runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`).
3. Run the enhancer with `demo.launch(share=True)` to get a free cloud-hosted link powered by Google's cloud GPU.
