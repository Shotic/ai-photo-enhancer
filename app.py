import time
import gradio as gr
from PIL import Image
from enhancer import ImageEnhancer

# Initialize model once at startup
enhancer = ImageEnhancer()

CUSTOM_CSS = """
/* Mobile-first optimizations for iOS Safari */
.gradio-container {
    max-width: 950px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

#header-title {
    text-align: center;
    margin-bottom: 0.5rem;
}

#header-desc {
    text-align: center;
    color: #64748b;
    margin-bottom: 1.5rem;
}

.action-btn {
    font-weight: 600 !important;
    border-radius: 12px !important;
    min-height: 48px !important; /* iOS touch target standard */
}

.stat-box {
    border-radius: 10px;
    padding: 10px 14px;
    background-color: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
}
"""

def apply_preset(preset_name):
    # Returns (scale, sharpness, vibrancy, contrast, skin_smoothing)
    if preset_name == "Wink Portrait HD (Face & Skin)":
        return "1x (Enhanced Original Size)", 1.15, 1.08, 1.04, 0.25
    elif preset_name == "Ultra Sharp 2x (Social Media)":
        return "2x", 1.25, 1.05, 1.05, 0.0
    elif preset_name == "Maximum Upscale 4x (Wallpaper / Print)":
        return "4x", 1.10, 1.02, 1.02, 0.0
    else: # Natural / Default
        return "2x", 1.05, 1.02, 1.00, 0.0

def process_image(
    input_img: Image.Image,
    scale: str,
    sharpness: float,
    vibrancy: float,
    contrast: float,
    skin_smoothing: float,
):
    if input_img is None:
        raise gr.Error("Please upload or take a photo first.")

    orig_w, orig_h = input_img.size
    start_time = time.time()

    enhanced = enhancer.enhance(
        image=input_img,
        target_scale=scale,
        sharpness_boost=sharpness,
        vibrancy_boost=vibrancy,
        contrast_boost=contrast,
        skin_smoothing=skin_smoothing,
    )

    elapsed = round(time.time() - start_time, 2)
    new_w, new_h = enhanced.size

    stats_html = f"""
    <div class="stat-box">
        <b>Success:</b> Processed in <b>{elapsed}s</b><br>
        <b>Original Dimensions:</b> {orig_w} &times; {orig_h} px<br>
        <b>Enhanced Dimensions:</b> {new_w} &times; {new_h} px ({scale})
    </div>
    """

    return enhanced, stats_html

with gr.Blocks(title="AI Photo Enhancer") as demo:
    gr.HTML("""
    <div id="header-title">
        <h1>✨ AI Photo Enhancer</h1>
    </div>
    <div id="header-desc">
        Free, open-source alternative to Wink & Remini. Enhances facial details, removes blur, sharpens textures, and upscales resolution.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Image (or Tap to Capture on iOS)",
                type="pil",
                sources=["upload", "webcam", "clipboard"]
            )

            preset_selector = gr.Radio(
                label="Quick Style Preset",
                choices=[
                    "Natural / Default",
                    "Wink Portrait HD (Face & Skin)",
                    "Ultra Sharp 2x (Social Media)",
                    "Maximum Upscale 4x (Wallpaper / Print)"
                ],
                value="Natural / Default",
                interactive=True
            )

            with gr.Accordion("Advanced Tuning Sliders", open=False):
                scale_dropdown = gr.Dropdown(
                    label="Upscale Factor",
                    choices=["1x (Enhanced Original Size)", "2x", "4x"],
                    value="2x",
                    interactive=True
                )
                sharpness_slider = gr.Slider(
                    minimum=0.8,
                    maximum=2.0,
                    value=1.05,
                    step=0.05,
                    label="Sharpness Boost",
                    interactive=True
                )
                skin_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Skin Softening Blend",
                    interactive=True
                )
                vibrancy_slider = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1.02,
                    step=0.02,
                    label="Color Vibrancy",
                    interactive=True
                )
                contrast_slider = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1.00,
                    step=0.02,
                    label="Contrast Boost",
                    interactive=True
                )

            enhance_btn = gr.Button("⚡ Enhance Photo", variant="primary", elem_classes=["action-btn"])

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Enhanced Photo (Long-press on iOS to Save)",
                type="pil",
                format="png"
            )
            stats_box = gr.HTML(visible=True)

    # Wire Preset Selector to automatically update slider values
    preset_selector.change(
        fn=apply_preset,
        inputs=[preset_selector],
        outputs=[scale_dropdown, sharpness_slider, vibrancy_slider, contrast_slider, skin_slider]
    )

    # Wire Enhance button click
    enhance_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            scale_dropdown,
            sharpness_slider,
            vibrancy_slider,
            contrast_slider,
            skin_slider
        ],
        outputs=[output_image, stats_box]
    )

if __name__ == "__main__":
    # Launch server with share=True to generate a free public HTTPS link for your iPhone
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS, share=True)


