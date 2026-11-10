import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageFilter
import threading
import torch
import os
import time
import random
import json
import re

from diffusers import AutoPipelineForText2Image
device = "cuda" if torch.cuda.is_available() else "cpu"

# THEME

ACCENT = "#7C5CFF"         
ACCENT_HOVER = "#6947E6"
BG_DARK = "#0F1117"
CARD = "#171A23"
CARD_LIGHT = "#1E2230"
TEXT_MUTED = "#8A8FA3"
DANGER = "#FF5C5C"
DANGER_HOVER = "#E64545"
SUCCESS = "#3DDC97"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_TITLE = ("Segoe UI Semibold", 22)
FONT_SUB = ("Segoe UI", 13)
FONT_LABEL = ("Segoe UI Semibold", 13)
FONT_BTN = ("Segoe UI Semibold", 13)
FONT_SMALL = ("Segoe UI", 11)

# FAST MODEL

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

generated_image = None
image_history = []


# DEEP-LEARNING FEATURE STATE (GUI layout remains unchanged)
uploaded_image = None
image_records = []          # Metadata for evaluation and comparison
last_generation_time = 0.0
last_prompt = ""
last_seed = None
last_steps = 4
last_guidance = 0.0
img2img_pipe = None
clip_model = None
clip_processor = None
upscale_pipe = None
caption_processor = None
caption_model = None
caption_cache = {}       # id(image) -> generated caption
caption_in_progress = set()
selected_image_id = None
denoising_visualizer_enabled = True

def _set_status_safe(message, color=TEXT_MUTED):
    """Schedule status changes on Tk's main thread."""
    try:
        app.after(0, lambda: set_status(message, color))
    except Exception:
        pass

def enhance_prompt_text(prompt):
    """Lightweight NLP-style prompt enrichment; no extra model download required."""
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    if not prompt:
        return prompt
    lower = prompt.lower()
    additions = []
    if not any(term in lower for term in ("detailed", "high detail", "intricate")):
        additions.append("highly detailed")
    if not any(term in lower for term in ("lighting", "lit", "sunlight", "moonlight", "neon")):
        additions.append("cinematic lighting")
    if not any(term in lower for term in ("composition", "close-up", "wide shot", "portrait", "macro")):
        additions.append("well-composed")
    if not any(term in lower for term in ("photorealistic", "watercolor", "anime", "oil painting", "3d render", "illustration")):
        additions.append("digital art")
    return prompt.rstrip(" .") + ", " + ", ".join(additions) + "."

def enhance_prompt_dialog():
    prompt = prompt_entry.get().strip()
    if not prompt:
        set_status("Enter a prompt to enhance first", "#FFB020")
        return
    enhanced = enhance_prompt_text(prompt)
    prompt_entry.delete(0, "end")
    prompt_entry.insert(0, enhanced)
    set_status("Prompt enhanced in the prompt field — review it, then generate", SUCCESS)

def choose_source_image():
    """Ctrl+O: choose an image for image-to-image generation."""
    global uploaded_image
    path = filedialog.askopenfilename(
        title="Choose source image for image-to-image generation",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")]
    )
    if not path:
        return
    try:
        uploaded_image = Image.open(path).convert("RGB")
        show_image(uploaded_image)
        set_status(f"Image-to-image source loaded: {os.path.basename(path)}", SUCCESS)
    except Exception as exc:
        messagebox.showerror("Image load error", str(exc))

def _get_img2img_pipeline():
    global img2img_pipe
    if img2img_pipe is None:
        from diffusers import AutoPipelineForImage2Image
        # Reuse the already-loaded SD-Turbo weights where supported.
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        img2img_pipe.to(device)
    return img2img_pipe

def _get_clip():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        clip_processor = CLIPProcessor.from_pretrained(model_id)
        clip_model = CLIPModel.from_pretrained(model_id).to(device)
        clip_model.eval()
    return clip_model, clip_processor

def calculate_clip_similarity(image, prompt):
    """Return CLIP cosine similarity as a percentage-like score (not a probability)."""
    model, processor = _get_clip()
    inputs = processor(text=[prompt], images=image.convert("RGB"), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = torch.sum(image_features * text_features, dim=-1).item()
    return round(score * 100, 2)

def _make_denoising_callback(active_pipe, total_steps):
    """Create a Diffusers callback that previews intermediate denoising states."""
    def on_step_end(pipeline, step_index, timestep, callback_kwargs):
        # Update progress in the existing Home status area on the Tk main thread.
        current = step_index + 1
        _set_status_safe(f"Diffusion denoising: step {current}/{total_steps} — refining image...", ACCENT)
        try:
            latents = callback_kwargs.get("latents")
            if latents is not None and current == 1 or (latents is not None and current == total_steps) or (latents is not None and current % 2 == 0):
                with torch.no_grad():
                    vae_latents = latents / active_pipe.vae.config.scaling_factor
                    decoded = active_pipe.vae.decode(vae_latents, return_dict=False)[0]
                    interim = active_pipe.image_processor.postprocess(decoded, output_type="pil")[0]
                # Capture a private copy for the UI callback.
                app.after(0, lambda im=interim.copy(), n=current: _show_denoising_preview(im, n, total_steps))
        except Exception:
            # A preview decoding issue should never abort actual image generation.
            pass
        return callback_kwargs
    return on_step_end

def _show_denoising_preview(image, step, total_steps):
    """Display a transient denoising snapshot in the existing image preview."""
    try:
        preview = ctk.CTkImage(light_image=image, dark_image=image, size=(360, 360))
        image_label.configure(image=preview, text="")
        image_label.image = preview
        set_status(f"Diffusion denoising: step {step}/{total_steps}", ACCENT)
    except Exception:
        pass

def _get_caption_model():
    """Load the pretrained BLIP image-captioning model on first use."""
    global caption_processor, caption_model
    if caption_processor is None or caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-base"
        caption_processor = BlipProcessor.from_pretrained(model_id)
        caption_model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        caption_model.eval()
    return caption_processor, caption_model


def generate_image_caption(image):
    """Generate a natural-language caption for one specific image."""
    processor, model = _get_caption_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=45, num_beams=3)
    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption or "No caption could be generated for this image."


def _start_caption_generation(image):
    """Caption each image asynchronously and update the caption for the selected image only."""
    image_key = id(image)
    if image_key in caption_cache or image_key in caption_in_progress:
        return
    caption_in_progress.add(image_key)

    def worker():
        try:
            caption = generate_image_caption(image)
            caption_cache[image_key] = caption
            def update_caption():
                caption_in_progress.discard(image_key)
                if selected_image_id == image_key:
                    caption_label.configure(text=f"Image caption: {caption}", text_color="#D7D9E2")
            app.after(0, update_caption)
        except Exception as exc:
            caption_in_progress.discard(image_key)
            error_text = str(exc).replace("\\n", " ")[:140]
            def show_caption_error():
                if selected_image_id == image_key:
                    caption_label.configure(
                        text=f"Caption unavailable: {error_text}",
                        text_color="#FFB020"
                    )
            app.after(0, show_caption_error)

    threading.Thread(target=worker, daemon=True).start()


def _display_caption_for_image(image):
    """Show the cached caption for the chosen gallery image, or start captioning it."""
    image_key = id(image)
    if image_key in caption_cache:
        caption_label.configure(text=f"Image caption: {caption_cache[image_key]}", text_color="#D7D9E2")
    else:
        caption_label.configure(
            text="Image caption: generating caption… (first use downloads the caption model)",
            text_color=TEXT_MUTED
        )
        _start_caption_generation(image)


def _generation_worker(prompt, count, seed_start, use_img2img, source_image, steps, guidance, strength):
    global generated_image, last_generation_time, last_prompt, last_seed, last_steps, last_guidance
    try:
        start_loading()
        started = time.time()
        results = []
        for offset in range(count):
            seed = seed_start + offset
            generator = torch.Generator(device=device).manual_seed(seed)
            if use_img2img and source_image is not None:
                active_pipe = _get_img2img_pipeline()
                image = active_pipe(
                    prompt=prompt,
                    image=source_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    strength=strength,
                    generator=generator,
                    callback_on_step_end=_make_denoising_callback(active_pipe, steps),
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images[0]
            else:
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=_make_denoising_callback(pipe, steps),
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images[0]
            results.append((image, seed))
        elapsed = time.time() - started
        last_generation_time = elapsed
        last_prompt = prompt
        last_seed = seed_start
        last_steps = steps
        last_guidance = guidance

        def update_ui():
            global generated_image
            for image, seed in results:
                generated_image = image
                image_history.append(image)
                image_records.append({
                    "prompt": prompt, "seed": seed, "steps": steps,
                    "guidance": guidance, "seconds": elapsed / max(1, len(results)),
                    "width": image.width, "height": image.height,
                    "clip_similarity": None
                })
            if results:
                # Start a separate caption job for every generated variation.
                for image, _seed in results:
                    _start_caption_generation(image)
                show_image(results[-1][0])
            update_gallery()
            stop_loading()
            set_status(f"Generated {len(results)} variation(s) in {elapsed:.1f}s", SUCCESS)
        app.after(0, update_ui)
    except Exception as exc:
        def show_error():
            stop_loading()
            set_status(f"Generation failed: {exc}", DANGER)
            messagebox.showerror("Generation error", str(exc))
        app.after(0, show_error)

def open_diffusion_lab():
    """A small settings dialog; the main application layout is untouched."""
    dialog = ctk.CTkToplevel(app)
    dialog.title("Diffusion Parameter Lab")
    dialog.geometry("440x390")
    dialog.transient(app)
    dialog.grab_set()

    ctk.CTkLabel(dialog, text="Diffusion Parameter Lab", font=FONT_TITLE).pack(pady=(18, 8))
    ctk.CTkLabel(dialog, text="Compare inference settings without changing the main GUI.",
                 font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 12))

    steps_var = ctk.IntVar(value=4)
    guidance_var = ctk.DoubleVar(value=0.0)
    count_var = ctk.IntVar(value=4)
    strength_var = ctk.DoubleVar(value=0.65)

    def add_slider(label, variable, start, end, number_of_steps=None):
        ctk.CTkLabel(dialog, text=label, font=FONT_LABEL).pack(anchor="w", padx=24)
        slider = ctk.CTkSlider(dialog, from_=start, to=end, variable=variable,
                               number_of_steps=number_of_steps)
        slider.pack(fill="x", padx=24, pady=(2, 8))
        return slider

    add_slider("Inference steps (1–8 recommended for SD-Turbo)", steps_var, 1, 8, 7)
    add_slider("Guidance scale (SD-Turbo works best near 0)", guidance_var, 0, 8, 80)
    add_slider("Number of variations (1–4)", count_var, 1, 4, 3)
    add_slider("Image-to-image strength (0.1–1.0)", strength_var, 0.1, 1.0, 18)

    def apply():
        app.lab_steps = int(steps_var.get())
        app.lab_guidance = float(guidance_var.get())
        app.lab_count = max(1, min(4, int(count_var.get())))
        app.lab_strength = float(strength_var.get())
        dialog.destroy()
        set_status(f"Lab settings saved: {app.lab_count} variation(s), {app.lab_steps} steps", SUCCESS)

    ctk.CTkButton(dialog, text="Apply settings", command=apply,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(pady=14)

def show_evaluation_dashboard():
    """Show generation metrics and optional CLIP scores in a separate window."""
    if not image_records:
        messagebox.showinfo("Evaluation Dashboard", "Generate at least one image first.")
        return
    win = ctk.CTkToplevel(app)
    win.title("Deep Learning Evaluation Dashboard")
    win.geometry("850x620")
    win.transient(app)
    ctk.CTkLabel(win, text="Deep Learning Evaluation Dashboard", font=FONT_TITLE).pack(pady=(16, 6))
    records = image_records[-50:]
    avg_time = sum(r["seconds"] for r in records) / len(records)
    ctk.CTkLabel(
        win,
        text=f"Images evaluated: {len(records)}    |    Average generation time: {avg_time:.2f}s/image",
        font=FONT_SUB
    ).pack(pady=4)

    table = ctk.CTkTextbox(win, height=250, font=("Consolas", 11))
    table.pack(fill="both", expand=True, padx=16, pady=12)
    table.insert("end", " # | Seed       | Size         | Steps | Guidance | Time(s) | CLIP similarity\n")
    table.insert("end", "-" * 82 + "\n")
    for i, r in enumerate(records, 1):
        clip_score = "--" if r.get("clip_similarity") is None else f'{r["clip_similarity"]:.2f}'
        table.insert("end", f'{i:2d} | {r["seed"]:<10} | {r["width"]}x{r["height"]:<8} | '
                             f'{r["steps"]:<5} | {r["guidance"]:<8.2f} | {r["seconds"]:<7.2f} | {clip_score}\n')
    table.configure(state="disabled")

    def export_metrics():
        path = filedialog.asksaveasfilename(
            title="Export evaluation metrics", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("JSON file", "*.json")]
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            else:
                import csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)
            set_status(f"Metrics exported: {os.path.basename(path)}", SUCCESS)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def show_chart():
        try:
            import matplotlib.pyplot as plt
            indices = list(range(1, len(records) + 1))
            times = [r["seconds"] for r in records]
            plt.figure(figsize=(8, 4.5))
            plt.plot(indices, times, marker="o")
            plt.title("Generation time by image")
            plt.xlabel("Image index")
            plt.ylabel("Seconds")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            messagebox.showerror("Chart error", f"Install matplotlib to view charts.\n{exc}")

    buttons = ctk.CTkFrame(win, fg_color="transparent")
    buttons.pack(pady=(0, 16))
    ctk.CTkButton(buttons, text="Plot generation time", command=show_chart).pack(side="left", padx=6)
    ctk.CTkButton(buttons, text="Export CSV / JSON", command=export_metrics).pack(side="left", padx=6)

def evaluate_current_image():
    if generated_image is None:
        set_status("Generate or select an image first", "#FFB020")
        return
    prompt = prompt_entry.get().strip() or last_prompt
    if not prompt:
        set_status("Enter the prompt used for this image", "#FFB020")
        return
    set_status("Calculating CLIP text-image similarity...", ACCENT)
    def worker():
        try:
            score = calculate_clip_similarity(generated_image, prompt)
            if image_records:
                image_records[-1]["clip_similarity"] = score
            _set_status_safe(f"CLIP similarity: {score:.2f} (relative score, not a probability)", SUCCESS)
        except Exception as exc:
            _set_status_safe(f"CLIP evaluation failed: {exc}", DANGER)
            app.after(0, lambda e=exc: messagebox.showerror("CLIP evaluation error", str(e)))
    threading.Thread(target=worker, daemon=True).start()

def super_resolve_current_image():
    """Run the pretrained Stable Diffusion x4 upscaler with compatible input sizing."""
    global upscale_pipe, generated_image
    source = generated_image.copy() if generated_image is not None else None
    if source is None:
        set_status("Generate or select an image first", "#FFB020")
        return
    if not messagebox.askyesno(
        "AI Super-Resolution",
        "Load the large pretrained Stable Diffusion x4 upscaler? This downloads a large model and may need substantial GPU memory."
    ):
        return

    def worker():
        global upscale_pipe, generated_image
        try:
            from diffusers import StableDiffusionUpscalePipeline
            if upscale_pipe is None:
                _set_status_safe("Loading super-resolution model (first use downloads it)...", ACCENT)
                upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                ).to(device)
                if device == "cuda":
                    try:
                        upscale_pipe.enable_attention_slicing()
                    except Exception:
                        pass

            prompt = prompt_entry.get().strip() or last_prompt or "high quality detailed image"
            source_img = source.convert("RGB")
            max_side = 256
            scale = min(1.0, max_side / max(source_img.size))
            if scale < 1.0:
                source_img = source_img.resize(
                    (max(64, int(source_img.width * scale)), max(64, int(source_img.height * scale))),
                    Image.Resampling.LANCZOS
                )
            if min(source_img.size) < 64:
                source_img = source_img.resize(
                    (max(64, source_img.width), max(64, source_img.height)), Image.Resampling.LANCZOS
                )

            _set_status_safe("Upscaling image with the diffusion model...", ACCENT)
            started = time.time()
            with torch.inference_mode():
                result = upscale_pipe(
                    prompt=prompt, image=source_img,
                    num_inference_steps=15, guidance_scale=7.0
                ).images[0]
            elapsed = time.time() - started

            def update():
                global generated_image
                generated_image = result
                image_history.append(result)
                image_records.append({
                    "prompt": prompt, "seed": 0, "steps": 15,
                    "guidance": 7.0, "seconds": elapsed, "width": result.width,
                    "height": result.height, "clip_similarity": None
                })
                show_image(result)
                update_gallery()
                set_status(f"AI super-resolution complete in {elapsed:.1f}s ({result.width}×{result.height})", SUCCESS)
            app.after(0, update)
        except Exception as exc:
            _set_status_safe(f"Super-resolution failed: {exc}", DANGER)
            app.after(0, lambda e=exc: messagebox.showerror(
                "Super-resolution error",
                f"{e}\\n\\nCheck updated diffusers/transformers/accelerate, internet access for model download, and available GPU/RAM."
            ))
    threading.Thread(target=worker, daemon=True).start()

def generate_variations():
    prompt = prompt_entry.get().strip()
    if not prompt:
        set_status("Enter a prompt first", "#FFB020")
        return
    count = getattr(app, "lab_count", 4)
    steps = getattr(app, "lab_steps", 4)
    guidance = getattr(app, "lab_guidance", 0.0)
    strength = getattr(app, "lab_strength", 0.65)
    seed_start = random.randint(0, 2**31 - count - 1)
    use_img2img = uploaded_image is not None
    source = uploaded_image.copy() if use_img2img else None
    threading.Thread(
        target=_generation_worker,
        args=(prompt, count, seed_start, use_img2img, source, steps, guidance, strength),
        daemon=True
    ).start()


# GENERATE IMAGE

def generate():
    generate_variations()




# SHOW IMAGE

def show_image(img):
    global selected_image_id
    selected_image_id = id(img)
    preview = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 360))
    image_label.configure(image=preview, text="")
    image_label.image = preview
    _display_caption_for_image(img)


# DELETE IMAGE

def delete_image(index):
    if 0 <= index < len(image_history):
        del image_history[index]
        update_gallery()


# SAVE ONE IMAGE 

def save():
    if generated_image is None:
        set_status("No image to save", "#FFB020")
        return

    file = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
        title="Save image to your computer"
    )
    if file:
        generated_image.save(file)
        set_status(f"Saved to {os.path.basename(file)}", SUCCESS)


# SAVE ALL IMAGES (LOCAL)

def save_all_images():
    if not image_history:
        set_status("No images to save", "#FFB020")
        return

    folder = filedialog.askdirectory(title="Choose a folder to save all images")
    if not folder:
        return

    for i, img in enumerate(image_history):
        img.save(os.path.join(folder, f"image_{i+1}.png"))

    set_status(f"Saved {len(image_history)} images", SUCCESS)

# STATUS HELPER

def set_status(text, color=TEXT_MUTED):
    status_label.configure(text=text, text_color=color)

# GALLERY (GRID)

def update_gallery():
    for widget in gallery_frame.winfo_children():
        widget.destroy()

    if not image_history:
        ctk.CTkLabel(
            gallery_frame, text="No images yet\nGenerate one to see it here",
            font=FONT_SMALL, text_color=TEXT_MUTED, justify="center"
        ).pack(pady=30)
        return

    cols = 2

    visible_images = image_history[-12:]
    start_index = len(image_history) - len(visible_images)
    for visible_index, img in enumerate(visible_images):
        index = start_index + visible_index
        row = index // cols
        col = index % cols

        thumb = ctk.CTkImage(light_image=img, dark_image=img, size=(95, 95))

        frame = ctk.CTkFrame(gallery_frame, fg_color=CARD_LIGHT, corner_radius=12)
        frame.grid(row=row, column=col, padx=6, pady=6)

        btn = ctk.CTkButton(
            frame,
            image=thumb,
            text="",
            width=95,
            height=95,
            corner_radius=10,
            fg_color="transparent",
            hover_color=CARD,
            command=lambda i=img: show_image(i)
        )
        btn.pack(padx=4, pady=4)

        del_btn = ctk.CTkButton(
            frame,
            text="✕",
            width=22,
            height=22,
            corner_radius=11,
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
            font=("Segoe UI", 11, "bold"),
            command=lambda i=index: delete_image(i)
        )
        del_btn.place(relx=1, rely=0, anchor="ne", x=-2, y=2)


# LOADING UI

def start_loading():
    """Update loading controls on Tk's main thread (generation runs in a worker)."""
    def update_controls():
        generate_btn.configure(state="disabled", text="Generating...")
        set_status("Creating your image — live diffusion denoising preview enabled...", ACCENT)
        spinner.set(0)
        spinner.start()
    try:
        app.after(0, update_controls)
    except Exception:
        pass


def stop_loading():
    spinner.stop()
    generate_btn.configure(state="normal", text="✨  Generate")
    set_status("Done — image ready", SUCCESS)


# SIDEBAR FUNCTIONS

def clear_image():
    global generated_image
    generated_image = None
    image_label.configure(image=None, text="Your generated image\nwill appear here")
    set_status("Cleared", TEXT_MUTED)


def close_app():
    if messagebox.askokcancel("Exit", "Do you really want to close the app?"):
        app.destroy()



# FEATURE TAB DASHBOARD
# The Home tab retains the original prompt bar, preview, and gallery design.
# Other features open in dedicated tabs with matching dark cards and controls.

def build_feature_tabs(parent):
    tabs = ctk.CTkTabview(
        parent,
        corner_radius=14,
        segmented_button_fg_color=CARD,
        segmented_button_selected_color=ACCENT,
        segmented_button_selected_hover_color=ACCENT_HOVER,
        segmented_button_unselected_color=CARD_LIGHT,
        segmented_button_unselected_hover_color="#2A2F42"
    )
    tabs.pack(fill="both", expand=True, padx=10, pady=10)

    tab_names = [
        "🏠 Home",
        "🖼 Image-to-Image",
        "🎲 Variations",
        "🔬 Diffusion Lab",
        "🎯 CLIP Evaluation",
        "⬆ Super-Resolution",
        "📊 Dashboard"
    ]
    for name in tab_names:
        tabs.add(name)

    # Shared page styling
    def page(tab, title, subtitle):
        wrap = ctk.CTkFrame(tab, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(wrap, text=title, font=FONT_TITLE, text_color="white").pack(anchor="w")
        ctk.CTkLabel(wrap, text=subtitle, font=FONT_SUB, text_color=TEXT_MUTED,
                     wraplength=780, justify="left").pack(anchor="w", pady=(4, 18))
        return wrap

    # The original GUI is reparented into the Home tab by constructing its content
    # there in the existing layout block below.
    p = page(tabs.tab("🖼 Image-to-Image"), "🖼 Image-to-Image Generation",
             "Load an image, enter a prompt on the Home tab, then transform the source image.")
    ctk.CTkButton(p, text="📂 Choose source image", command=choose_source_image,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w", pady=(0, 10))
    ctk.CTkButton(p, text="✨ Transform uploaded image", command=_generate_img2img_from_tab,
                  fg_color=CARD_LIGHT, hover_color="#2A2F42").pack(anchor="w")
    ctk.CTkLabel(p, text="The selected source image will appear in the existing preview.",
                 font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w", pady=12)

    p = page(tabs.tab("🎲 Variations"), "🎲 Multiple Image Variations",
             "Generate several outputs with distinct, recorded random seeds.")
    ctk.CTkLabel(p, text="Each variation is added to the existing gallery. Change the count and sampling settings in Diffusion Lab.",
                 font=FONT_SUB, text_color=TEXT_MUTED, wraplength=760, justify="left").pack(anchor="w", pady=(0, 14))
    ctk.CTkButton(p, text="Generate variations", command=generate_variations,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w")

    p = page(tabs.tab("🔬 Diffusion Lab"), "🔬 Diffusion Parameter Lab",
             "Control inference steps, guidance scale, variation count, and image-to-image strength.")
    ctk.CTkButton(p, text="Open parameter controls", command=open_diffusion_lab,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w")
    ctk.CTkLabel(p, text="SD-Turbo is designed for very few inference steps and guidance near zero. Use other values for controlled experiments.",
                 font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=760, justify="left").pack(anchor="w", pady=12)

    p = page(tabs.tab("🎯 CLIP Evaluation"), "🎯 Text–Image Similarity",
             "Use a pretrained CLIP vision-language model to estimate how closely the current image aligns with a text prompt.")
    ctk.CTkButton(p, text="Calculate CLIP similarity", command=evaluate_current_image,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w")
    ctk.CTkLabel(p, text="The score is a relative similarity measure, not a probability or a complete quality score.",
                 font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=760, justify="left").pack(anchor="w", pady=12)

    p = page(tabs.tab("⬆ Super-Resolution"), "⬆ AI Super-Resolution",
             "Upscale the current image using a pretrained diffusion super-resolution model.")
    ctk.CTkButton(p, text="Upscale current image", command=super_resolve_current_image,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w")
    ctk.CTkLabel(p, text="This downloads a separate large model the first time and may require substantial GPU memory.",
                 font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=760, justify="left").pack(anchor="w", pady=12)

    p = page(tabs.tab("📊 Dashboard"), "📊 Evaluation Dashboard",
             "Review generation times, seeds, image dimensions, inference settings, CLIP similarity scores, and export metrics.")
    ctk.CTkButton(p, text="Open metrics and charts", command=show_evaluation_dashboard,
                  fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w")
    ctk.CTkLabel(p, text="Export recorded metrics as CSV or JSON for your mini-project report.",
                 font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w", pady=12)

    return tabs

def _enhance_from_box(textbox):
    value = textbox.get("1.0", "end").strip()
    if not value:
        value = prompt_entry.get().strip()
    if not value:
        set_status("Enter a prompt in the Home tab or NLP tab", "#FFB020")
        return
    enhanced = enhance_prompt_text(value)
    textbox.delete("1.0", "end")
    textbox.insert("1.0", enhanced)
    prompt_entry.delete(0, "end")
    prompt_entry.insert(0, enhanced)
    set_status("Enhanced prompt copied to Home tab", SUCCESS)

def _generate_img2img_from_tab():
    if uploaded_image is None:
        choose_source_image()
    if uploaded_image is None:
        return
    generate_variations()


# UI SETUP

app = ctk.CTk()
app.geometry("1150x640")
app.minsize(950, 560)
app.title("AI Studio — Deep Learning Image Studio")
app.configure(fg_color=BG_DARK)

app.protocol("WM_DELETE_WINDOW", close_app)

# Keyboard shortcuts keep the existing GUI layout unchanged:
# Ctrl+O load image for image-to-image; Ctrl+P enhance prompt;
# Ctrl+L diffusion parameter lab; Ctrl+E CLIP evaluation;
# Ctrl+D evaluation dashboard; Ctrl+U AI super-resolution.
app.lab_steps = 4
app.lab_guidance = 0.0
app.lab_count = 4
app.lab_strength = 0.65
app.bind("<Control-o>", lambda event: choose_source_image())
app.bind("<Control-p>", lambda event: enhance_prompt_dialog())
app.bind("<Control-l>", lambda event: open_diffusion_lab())
app.bind("<Control-e>", lambda event: evaluate_current_image())
app.bind("<Control-d>", lambda event: show_evaluation_dashboard())
app.bind("<Control-u>", lambda event: super_resolve_current_image())

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# SIDEBAR

sidebar = ctk.CTkFrame(app, width=220, fg_color=CARD, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nsw")
sidebar.grid_propagate(False)

logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
logo_frame.pack(pady=(28, 6), padx=20, fill="x")

ctk.CTkLabel(logo_frame, text="✨ AI Studio", font=FONT_TITLE, text_color="white").pack(anchor="w")
ctk.CTkLabel(logo_frame, text="Local image generator", font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w")

ctk.CTkFrame(sidebar, height=1, fg_color="#2A2E3D").pack(fill="x", padx=20, pady=18)

nav_label = ctk.CTkLabel(sidebar, text="ACTIONS", font=("Segoe UI", 10, "bold"), text_color=TEXT_MUTED)
nav_label.pack(anchor="w", padx=22, pady=(0, 8))

def sidebar_button(text, command, fg=CARD_LIGHT, hover="#2A2F42", text_color="white"):
    return ctk.CTkButton(
        sidebar, text=text, command=command,
        font=FONT_BTN, fg_color=fg, hover_color=hover, text_color=text_color,
        corner_radius=10, height=42, anchor="w"
    )

sidebar_button("🗑  Clear Preview", clear_image).pack(pady=6, padx=20, fill="x")
sidebar_button("💾  Save Image", save, fg=ACCENT, hover=ACCENT_HOVER).pack(pady=6, padx=20, fill="x")
sidebar_button("📁  Save All to Folder", save_all_images).pack(pady=6, padx=20, fill="x")

sidebar_button("🖼  Image-to-Image", lambda: tabs.set("🖼 Image-to-Image")).pack(pady=4, padx=20, fill="x")
sidebar_button("🎲  Variations", lambda: tabs.set("🎲 Variations")).pack(pady=4, padx=20, fill="x")
sidebar_button("🔬  Diffusion Lab", lambda: tabs.set("🔬 Diffusion Lab")).pack(pady=4, padx=20, fill="x")
sidebar_button("🎯  CLIP Evaluation", lambda: tabs.set("🎯 CLIP Evaluation")).pack(pady=4, padx=20, fill="x")
sidebar_button("⬆  Super-Resolution", lambda: tabs.set("⬆ Super-Resolution")).pack(pady=4, padx=20, fill="x")
sidebar_button("📊  Dashboard", lambda: tabs.set("📊 Dashboard")).pack(pady=4, padx=20, fill="x")


bottom_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
bottom_frame.pack(side="bottom", fill="x", padx=20, pady=24)

ctk.CTkLabel(bottom_frame, text=f"Device: {device.upper()}", font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 10))

sidebar_button("⏻  Exit", close_app, fg=DANGER, hover=DANGER_HOVER).pack(in_=bottom_frame, fill="x")

# MAIN AREA

tabs_host = ctk.CTkFrame(app, fg_color="transparent")
tabs_host.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
tabs = build_feature_tabs(tabs_host)
home_tab = tabs.tab("🏠 Home")
main_frame = ctk.CTkFrame(home_tab, fg_color="transparent")
main_frame.pack(fill="both", expand=True, padx=12, pady=12)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure(2, weight=1)

# Header
header = ctk.CTkFrame(main_frame, fg_color="transparent")
header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
ctk.CTkLabel(header, text="Generate an image", font=FONT_TITLE, text_color="white").pack(anchor="w")
ctk.CTkLabel(header, text="Describe what you want to see, then generate and save it locally.",
             font=FONT_SUB, text_color=TEXT_MUTED).pack(anchor="w", pady=(2, 0))

# Prompt bar (card)
prompt_card = ctk.CTkFrame(main_frame, fg_color=CARD, corner_radius=16)
prompt_card.grid(row=1, column=0, sticky="ew", pady=(0, 20))
prompt_card.grid_columnconfigure(0, weight=1)

prompt_inner = ctk.CTkFrame(prompt_card, fg_color="transparent")
prompt_inner.pack(fill="x", padx=18, pady=16)
prompt_inner.grid_columnconfigure(0, weight=1)

prompt_entry = ctk.CTkEntry(
    prompt_inner, placeholder_text="A cinematic portrait of a fox in a neon city, 4k...",
    font=FONT_SUB, height=44, corner_radius=12, fg_color=CARD_LIGHT, border_width=0
)
prompt_entry.grid(row=0, column=0, sticky="ew", padx=(0, 12))
prompt_entry.bind("<Return>", lambda e: generate())

generate_btn = ctk.CTkButton(
    prompt_inner, text="✨  Generate", command=generate,
    font=FONT_BTN, height=44, corner_radius=12, width=140,
    fg_color=ACCENT, hover_color=ACCENT_HOVER
)
generate_btn.grid(row=0, column=2)

enhance_btn = ctk.CTkButton(
    prompt_inner, text="✨ Enhance", command=enhance_prompt_dialog,
    font=FONT_BTN, height=44, corner_radius=12, width=120,
    fg_color=CARD_LIGHT, hover_color="#2A2F42"
)
enhance_btn.grid(row=0, column=1, padx=(0, 10))

status_row = ctk.CTkFrame(prompt_card, fg_color="transparent")
status_row.pack(fill="x", padx=18, pady=(0, 14))

status_label = ctk.CTkLabel(status_row, text="Ready", font=FONT_SMALL, text_color=TEXT_MUTED)
status_label.pack(side="left")

spinner = ctk.CTkProgressBar(status_row, mode="indeterminate", height=4, fg_color=CARD_LIGHT, progress_color=ACCENT)
spinner.pack(side="left", fill="x", expand=True, padx=(14, 0))

# Content: preview + gallery
content = ctk.CTkFrame(main_frame, fg_color="transparent")
content.grid(row=2, column=0, sticky="nsew")
content.grid_columnconfigure(0, weight=3)
content.grid_columnconfigure(1, weight=1)
content.grid_rowconfigure(0, weight=1)

# Preview card
preview_card = ctk.CTkFrame(content, fg_color=CARD, corner_radius=16)
preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

image_label = ctk.CTkLabel(
    preview_card, text="Your generated image\nwill appear here",
    font=FONT_SUB, text_color=TEXT_MUTED, justify="center"
)
image_label.place(relx=0.5, rely=0.46, anchor="center")

caption_label = ctk.CTkLabel(
    preview_card,
    text="Image caption will appear here after generation. Click a gallery image to view its caption.",
    font=FONT_SMALL,
    text_color=TEXT_MUTED,
    wraplength=560,
    justify="left"
)
caption_label.place(relx=0.04, rely=0.94, anchor="sw")

# Gallery card
gallery_card = ctk.CTkFrame(content, fg_color=CARD, corner_radius=16)
gallery_card.grid(row=0, column=1, sticky="nsew")
gallery_card.grid_rowconfigure(1, weight=1)
gallery_card.grid_columnconfigure(0, weight=1)

gallery_header = ctk.CTkFrame(gallery_card, fg_color="transparent")
gallery_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 4))
ctk.CTkLabel(gallery_header, text="Gallery", font=FONT_LABEL, text_color="white").pack(anchor="w")

gallery_frame = ctk.CTkScrollableFrame(gallery_card, fg_color="transparent")
gallery_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 12))

update_gallery()

app.mainloop()