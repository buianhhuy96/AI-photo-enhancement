"""
WindowSeat Reflection Removal - Desktop Application.
Topaz-style UI with side panel controls and progress popup.

Usage:
    python app.py          # Full mode (requires GPU + model)
    python app.py --mock   # UI preview mode (no model needed)
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

from backend.config import InferenceParams, SUPPORTED_EXTENSIONS, params_from_quality

# Parse args
_parser = argparse.ArgumentParser()
_parser.add_argument("--mock", action="store_true", help="Run UI without AI model")
_parser.add_argument("--web", action="store_true", help="Launch as web app (accessible via browser)")
_parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
_parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
_args, _ = _parser.parse_known_args()
MOCK_MODE = _args.mock
WEB_MODE = _args.web

if not MOCK_MODE:
    from backend.engine import WindowSeatEngine, collect_images
    engine = WindowSeatEngine()
else:
    engine = None

    def collect_images(path):
        p = Path(path)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if p.is_file() and p.suffix.lower() in exts:
            return [str(p)]
        elif p.is_dir():
            return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)
        return []


# ============================================================
#  Theme & Appearance
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PANEL_WIDTH = 320
ACCENT = "#6366f1"
BG_DARK = "#1a1a2e"
BG_PANEL = "#16213e"
BG_CARD = "#0f3460"


# ============================================================
#  Progress Popup
# ============================================================
class ProgressPopup(ctk.CTkToplevel):
    """Modal progress dialog centered over main window."""

    def __init__(self, parent, title="Processing..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("450x200")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Center over parent
        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - 450) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 200) // 2
        self.geometry(f"+{px}+{py}")

        self.configure(fg_color=BG_DARK)

        self.status_label = ctk.CTkLabel(self, text="Initializing...",
                                         font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=(30, 10))

        self.progress_bar = ctk.CTkProgressBar(self, width=380)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.detail_label = ctk.CTkLabel(self, text="",
                                         font=ctk.CTkFont(size=12),
                                         text_color="gray")
        self.detail_label.pack(pady=5)

        self.cancel_btn = ctk.CTkButton(self, text="Cancel", width=100,
                                        fg_color="gray", command=self._cancel)
        self.cancel_btn.pack(pady=10)

        self.cancelled = False

    def update_progress(self, fraction, status="", detail=""):
        self.progress_bar.set(fraction)
        if status:
            self.status_label.configure(text=status)
        if detail:
            self.detail_label.configure(text=detail)
        self.update()

    def _cancel(self):
        self.cancelled = True
        self.status_label.configure(text="Cancelling...")

    def close(self):
        self.grab_release()
        self.destroy()


# ============================================================
#  Main Application
# ============================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WindowSeat — Reflection Removal")
        self.geometry("1280x780")
        self.minsize(1000, 600)
        self.configure(fg_color=BG_DARK)

        self.input_path = None
        self.output_dir = None
        self.input_images = []
        self.current_preview_idx = 0
        self.original_image = None
        self.result_image = None

        self._build_layout()
        self._update_gpu_status()

    # ----------------------------------------------------------
    #  Layout
    # ----------------------------------------------------------
    def _build_layout(self):
        # Main horizontal split: content (left) | panel (right)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # --- Content area ---
        self.content_frame = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.content_frame.grid_rowconfigure(1, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(header, text="WindowSeat",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        self.gpu_label = ctk.CTkLabel(header, text="GPU: detecting...",
                                      font=ctk.CTkFont(size=11),
                                      text_color="gray")
        self.gpu_label.pack(side="right", padx=10)

        if MOCK_MODE:
            ctk.CTkLabel(header, text="MOCK MODE",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#fbbf24").pack(side="right", padx=10)

        # Image panels
        self.input_panel = self._create_image_panel(self.content_frame, "Original")
        self.input_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.output_panel = self._create_image_panel(self.content_frame, "Result")
        self.output_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # --- Right panel (controls) ---
        self.panel = ctk.CTkScrollableFrame(self, width=PANEL_WIDTH,
                                            fg_color=BG_PANEL,
                                            corner_radius=0)
        self.panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        self._build_panel()

    def _create_image_panel(self, parent, label_text):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(frame, text=label_text,
                             font=ctk.CTkFont(size=12, weight="bold"),
                             text_color="gray")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

        canvas_frame = ctk.CTkFrame(frame, fg_color="transparent")
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        img_label = ctk.CTkLabel(canvas_frame, text="No image loaded",
                                 text_color="gray")
        img_label.grid(row=0, column=0, sticky="nsew")

        frame._img_label = img_label
        frame._photo_ref = None
        return frame

    def _set_panel_image(self, panel, pil_image):
        """Display a PIL image fitted to panel size."""
        panel.update_idletasks()
        pw = max(panel.winfo_width() - 20, 200)
        ph = max(panel.winfo_height() - 50, 200)

        img = pil_image.copy()
        img.thumbnail((pw, ph), Image.LANCZOS)

        photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        panel._img_label.configure(image=photo, text="")
        panel._photo_ref = photo

    # ----------------------------------------------------------
    #  Right Panel Controls
    # ----------------------------------------------------------
    def _build_panel(self):
        p = self.panel

        # --- Input section ---
        self._section_label(p, "INPUT")

        self.input_btn = ctk.CTkButton(p, text="Select Image / Folder",
                                       command=self._select_input,
                                       fg_color=ACCENT, hover_color="#4f46e5")
        self.input_btn.pack(fill="x", padx=10, pady=(5, 2))

        self.input_info = ctk.CTkLabel(p, text="No input selected",
                                       font=ctk.CTkFont(size=11),
                                       text_color="gray", wraplength=280)
        self.input_info.pack(fill="x", padx=10, pady=(0, 10))

        # --- Output section ---
        self._section_label(p, "OUTPUT")

        self.output_btn = ctk.CTkButton(p, text="Select Output Folder",
                                        command=self._select_output,
                                        fg_color="#374151", hover_color="#4b5563")
        self.output_btn.pack(fill="x", padx=10, pady=(5, 2))

        self.output_info = ctk.CTkLabel(p, text="Auto: input_dir/windowseat_output",
                                        font=ctk.CTkFont(size=11),
                                        text_color="gray", wraplength=280)
        self.output_info.pack(fill="x", padx=10, pady=(0, 10))

        # --- Separator ---
        ctk.CTkFrame(p, height=1, fg_color="gray").pack(fill="x", padx=10, pady=10)

        # --- Processing Parameters ---
        self._section_label(p, "PARAMETERS")

        # Quality slider
        ctk.CTkLabel(p, text="Quality", font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=10, pady=(8, 0))
        self.quality_var = ctk.IntVar(value=2)
        self.quality_slider = ctk.CTkSlider(p, from_=1, to=5,
                                            number_of_steps=4,
                                            variable=self.quality_var,
                                            command=self._on_quality_change)
        self.quality_slider.pack(fill="x", padx=10, pady=(2, 0))
        self.quality_label = ctk.CTkLabel(p, text="2 — Balanced",
                                          font=ctk.CTkFont(size=11),
                                          text_color="gray")
        self.quality_label.pack(anchor="w", padx=10, pady=(0, 5))

        # Strength slider
        ctk.CTkLabel(p, text="Strength", font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=10, pady=(8, 0))
        self.strength_var = ctk.DoubleVar(value=1.0)
        self.strength_slider = ctk.CTkSlider(p, from_=0.0, to=1.0,
                                             variable=self.strength_var,
                                             command=self._on_strength_change)
        self.strength_slider.pack(fill="x", padx=10, pady=(2, 0))
        self.strength_label = ctk.CTkLabel(p, text="1.00 — Full removal",
                                           font=ctk.CTkFont(size=11),
                                           text_color="gray")
        self.strength_label.pack(anchor="w", padx=10, pady=(0, 5))

        # --- Separator ---
        ctk.CTkFrame(p, height=1, fg_color="gray").pack(fill="x", padx=10, pady=10)

        # --- Advanced ---
        self._section_label(p, "ADVANCED")

        # 4-bit quantization
        self.use_4bit_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(p, text="4-bit quantization (saves VRAM)",
                        variable=self.use_4bit_var,
                        font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=5)

        # Output format
        ctk.CTkLabel(p, text="Output format", font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=10, pady=(8, 2))
        self.format_var = ctk.StringVar(value="png")
        format_frame = ctk.CTkFrame(p, fg_color="transparent")
        format_frame.pack(fill="x", padx=10)
        for fmt in ["png", "jpg", "webp"]:
            ctk.CTkRadioButton(format_frame, text=fmt.upper(),
                               variable=self.format_var, value=fmt,
                               font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 15))

        # JPEG quality
        ctk.CTkLabel(p, text="JPEG/WebP quality", font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=10, pady=(10, 0))
        self.jpg_quality_var = ctk.IntVar(value=95)
        self.jpg_quality_slider = ctk.CTkSlider(p, from_=50, to=100,
                                                variable=self.jpg_quality_var,
                                                command=self._on_jpg_quality_change)
        self.jpg_quality_slider.pack(fill="x", padx=10, pady=(2, 0))
        self.jpg_quality_label = ctk.CTkLabel(p, text="95",
                                              font=ctk.CTkFont(size=11),
                                              text_color="gray")
        self.jpg_quality_label.pack(anchor="w", padx=10, pady=(0, 5))

        # --- Separator ---
        ctk.CTkFrame(p, height=1, fg_color="gray").pack(fill="x", padx=10, pady=15)

        # --- Start Button ---
        self.start_btn = ctk.CTkButton(
            p, text="▶  Start Processing", height=48,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT, hover_color="#4f46e5",
            command=self._start_processing,
        )
        self.start_btn.pack(fill="x", padx=10, pady=(5, 10))

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#9ca3af").pack(anchor="w", padx=10, pady=(12, 2))

    # ----------------------------------------------------------
    #  Callbacks
    # ----------------------------------------------------------
    def _on_quality_change(self, val):
        v = int(round(val))
        labels = {1: "Fast", 2: "Balanced", 3: "High", 4: "Very High", 5: "Maximum"}
        self.quality_label.configure(text=f"{v} — {labels[v]}")

    def _on_strength_change(self, val):
        v = float(val)
        if v >= 0.95:
            desc = "Full removal"
        elif v >= 0.7:
            desc = "Strong"
        elif v >= 0.4:
            desc = "Moderate"
        elif v >= 0.1:
            desc = "Subtle"
        else:
            desc = "Off (original)"
        self.strength_label.configure(text=f"{v:.2f} — {desc}")

    def _on_jpg_quality_change(self, val):
        self.jpg_quality_label.configure(text=str(int(round(val))))

    def _select_input(self):
        choice = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if choice:
            self.input_images = list(choice)
            self.input_path = os.path.dirname(choice[0])
            self.input_info.configure(
                text=f"{len(self.input_images)} image(s) selected")
            self.current_preview_idx = 0
            self._show_input_preview(self.input_images[0])
            return

        # If cancelled file dialog, offer folder selection
        folder = filedialog.askdirectory(title="Or select a folder with images")
        if folder:
            self.input_path = folder
            self.input_images = collect_images(folder)
            self.input_info.configure(
                text=f"{len(self.input_images)} image(s) in\n{folder}")
            if self.input_images:
                self._show_input_preview(self.input_images[0])

    def _select_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir = folder
            self.output_info.configure(text=folder)

    def _show_input_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self.original_image = img
            self._set_panel_image(self.input_panel, img)
        except Exception as e:
            self.input_info.configure(text=f"Error loading: {e}")

    # ----------------------------------------------------------
    #  Processing
    # ----------------------------------------------------------
    def _start_processing(self):
        if not self.input_images:
            messagebox.showwarning("No Input", "Please select image(s) or a folder first.")
            return

        # Determine output directory
        output_dir = self.output_dir
        if not output_dir:
            if os.path.isfile(self.input_path):
                output_dir = os.path.join(os.path.dirname(self.input_path), "windowseat_output")
            else:
                output_dir = os.path.join(self.input_path, "windowseat_output")

        os.makedirs(output_dir, exist_ok=True)

        # Gather params
        params = params_from_quality(
            quality=self.quality_var.get(),
            use_4bit=self.use_4bit_var.get(),
            output_format=self.format_var.get(),
            jpg_quality=self.jpg_quality_var.get(),
        )
        strength = self.strength_var.get()

        # Show progress popup
        popup = ProgressPopup(self, title="Processing Images")

        # Run in background thread
        thread = threading.Thread(
            target=self._process_thread,
            args=(self.input_images, output_dir, params, strength, popup),
            daemon=True,
        )
        thread.start()

    def _process_thread(self, images, output_dir, params, strength, popup):
        """Background processing thread."""
        total = len(images)

        try:
            # Auto-initialize on first run
            if not MOCK_MODE and not engine._initialized:
                self.after(0, lambda: popup.update_progress(
                    0.0, "Loading model (first time, ~30-60s)...", "Downloading/loading weights"))
                engine.initialize()

            for idx, img_path in enumerate(images):
                if popup.cancelled:
                    self.after(0, popup.close)
                    return

                name = Path(img_path).stem
                ext = f".{params.output_format}"
                out_path = os.path.join(output_dir, f"{name}_clean{ext}")

                # Skip existing
                if os.path.exists(out_path):
                    self.after(0, lambda i=idx, n=name: popup.update_progress(
                        (i + 1) / total, f"Skipped ({n})", f"{i+1}/{total}"))
                    continue

                self.after(0, lambda i=idx, n=name: popup.update_progress(
                    i / total, f"Processing: {n}", f"Image {i+1}/{total}"))

                if MOCK_MODE:
                    for step in range(10):
                        if popup.cancelled:
                            break
                        time.sleep(0.2)
                        frac = (idx + (step + 1) / 10) / total
                        self.after(0, lambda f=frac, s=step: popup.update_progress(
                            f, detail=f"Step {s+1}/10"))
                    img = Image.open(img_path).convert("RGB")
                    img.save(out_path)
                else:
                    def progress_cb(msg, frac, _idx=idx):
                        overall = (_idx + frac) / total
                        self.after(0, lambda o=overall, m=msg: popup.update_progress(o, detail=m))

                    engine.process_image(img_path, out_path, params, progress_cb)

                    # Apply strength blending if < 1.0
                    if strength < 1.0:
                        original = Image.open(img_path).convert("RGB")
                        processed = Image.open(out_path).convert("RGB")
                        blended = Image.blend(original, processed, strength)
                        blended.save(out_path)

                # Update result preview
                self.after(0, lambda p=out_path: self._show_result(p))

            self.after(0, lambda: popup.update_progress(1.0, "Done!", f"All {total} images processed"))
            time.sleep(1)
            self.after(0, popup.close)
            self.after(0, lambda: messagebox.showinfo(
                "Complete", f"Processed {total} image(s)\nSaved to: {output_dir}"))

        except Exception as e:
            self.after(0, popup.close)
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _show_result(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self.result_image = img
            self._set_panel_image(self.output_panel, img)
        except Exception:
            pass

    # ----------------------------------------------------------
    #  GPU Status
    # ----------------------------------------------------------
    def _update_gpu_status(self):
        if MOCK_MODE:
            self.gpu_label.configure(text="MOCK — No GPU required")
            return
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / 1e9
                self.gpu_label.configure(text=f"GPU: {name} ({vram:.0f}GB)")
            else:
                self.gpu_label.configure(text="No CUDA GPU detected", text_color="#ef4444")
        except Exception:
            self.gpu_label.configure(text="GPU: unknown")


# ============================================================
#  Web UI (Gradio) — launched with --web flag
# ============================================================
def launch_web_ui():
    """Launch browser-accessible Gradio UI."""
    import gradio as gr

    def web_process_single(input_image, quality, strength, use_4bit, output_format, jpg_quality, progress=gr.Progress()):
        """Run AI processing, return processed result + store in state."""
        if input_image is None:
            return None, None, None, "No image provided."

        if MOCK_MODE:
            for i in range(10):
                time.sleep(0.3)
                progress(i / 10, desc=f"[MOCK] Step {i+1}/10")
            processed = input_image.copy()
            preview = Image.blend(input_image, processed, strength)
            return preview, input_image, processed, "Done — adjust Strength for live preview."

        if not engine._initialized:
            progress(0.0, desc="Loading model (~30-60s)...")
            engine.initialize()

        params = params_from_quality(int(quality), use_4bit, output_format, int(jpg_quality))
        tmp_dir = Path("temp_processing")
        tmp_dir.mkdir(exist_ok=True)
        input_path = str(tmp_dir / "input.png")
        input_image.save(input_path)
        output_path = str(tmp_dir / f"output.{output_format}")

        def progress_cb(msg, frac):
            progress(frac, desc=msg)

        try:
            engine.process_image(input_path, output_path, params, progress_cb)
            processed = Image.open(output_path).convert("RGB")
            preview = Image.blend(input_image, processed, strength)
            return preview, input_image, processed, "Done — adjust Strength for live preview."
        except Exception as e:
            return None, None, None, f"Error: {e}"

    def update_strength_preview(strength, original_state, processed_state):
        """Instant blend when strength slider changes."""
        if original_state is None or processed_state is None:
            return None
        return Image.blend(original_state, processed_state, strength)

    def export_single(strength, original_state, processed_state, output_format, jpg_quality):
        """Export current image at full quality."""
        if original_state is None or processed_state is None:
            return "Nothing to export. Process an image first."
        final = Image.blend(original_state, processed_state, strength)
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        out_path = export_dir / f"windowseat_{timestamp}.{output_format}"
        save_kwargs = {}
        if output_format in ("jpg", "jpeg"):
            save_kwargs["quality"] = int(jpg_quality)
        elif output_format == "webp":
            save_kwargs["quality"] = int(jpg_quality)
        final.save(str(out_path), **save_kwargs)
        return f"Exported → {out_path}"

    def export_all(image_list, quality, strength, use_4bit, output_format, jpg_quality, progress=gr.Progress()):
        """Export/process all imported images."""
        if not image_list:
            return "No images imported."
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        params = params_from_quality(int(quality), use_4bit, output_format, int(jpg_quality))
        lines = []
        for idx, img_data in enumerate(image_list):
            img = Image.open(img_data).convert("RGB") if isinstance(img_data, str) else img_data
            name = f"image_{idx+1}"
            if isinstance(img_data, str):
                name = Path(img_data).stem

            if MOCK_MODE:
                time.sleep(0.3)
                progress((idx + 1) / len(image_list), desc=f"[MOCK] {name}")
                out_path = export_dir / f"{name}_clean.{output_format}"
                img.save(str(out_path))
                lines.append(f"[{idx+1}/{len(image_list)}] {name}")
                continue

            if not engine._initialized:
                progress(0.0, desc="Loading model...")
                engine.initialize()

            tmp_dir = Path("temp_processing")
            tmp_dir.mkdir(exist_ok=True)
            input_path = str(tmp_dir / "batch_input.png")
            img.save(input_path)
            out_path = export_dir / f"{name}_clean.{output_format}"

            def pcb(msg, frac, _i=idx, _t=len(image_list)):
                progress((_i + frac) / _t, desc=f"[{_i+1}/{_t}] {msg}")
            try:
                engine.process_image(input_path, str(out_path), params, pcb)
                if strength < 1.0:
                    proc = Image.open(str(out_path)).convert("RGB")
                    Image.blend(img, proc, strength).save(str(out_path))
                lines.append(f"[{idx+1}/{len(image_list)}] Done: {name}")
            except Exception as e:
                lines.append(f"[{idx+1}/{len(image_list)}] Error: {name} — {e}")
        return f"Exported {len(image_list)} images → {export_dir}\n\n" + "\n".join(lines)

    def on_images_uploaded(file_list):
        """When files are uploaded, return count info and first image preview."""
        if not file_list:
            return None, "No images imported."
        first_img = Image.open(file_list[0]).convert("RGB")
        count = len(file_list)
        names = [Path(f).name for f in file_list[:5]]
        info = f"{count} image{'s' if count > 1 else ''} imported"
        if count > 5:
            info += f"\n{', '.join(names)}... +{count-5} more"
        else:
            info += f"\n{', '.join(names)}"
        return first_img, info

    def process_preview(file_list, quality, strength, use_4bit, output_format, jpg_quality, progress=gr.Progress()):
        """Process first image in batch as preview."""
        if not file_list:
            return None, None, None, "No images imported."
        first_img = Image.open(file_list[0]).convert("RGB")
        return web_process_single(first_img, quality, strength, use_4bit, output_format, jpg_quality, progress)


    # ================================================================
    # CSS
    # ================================================================
    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 0 !important;
        background: #1d1d1d !important;
    }
    body, html, .main, .app { background: #1d1d1d !important; }
    footer { display: none !important; }

    /* === Top bar === */
    .top-bar {
        background: #2d2d2d;
        border-bottom: 1px solid #3d3d3d;
        padding: 10px 24px;
    }
    .top-bar h2 { color: #999; font-size: 13px; font-weight: 400; margin: 0; }
    .top-bar span.accent { color: #e0e0e0; font-weight: 600; }

    /* === Remove defaults === */
    .group, .gr-group, .panel { background: transparent !important; border: none !important; box-shadow: none !important; }
    .block, .form { background: transparent !important; border: none !important; }

    /* === Left sidebar === */
    .sidebar {
        background: #252525 !important;
        border: none !important;
        border-right: 1px solid #3d3d3d !important;
        border-radius: 0 !important;
        padding: 12px 0 !important;
        min-height: calc(100vh - 50px) !important;
    }
    .sidebar .block { padding: 0 !important; }

    /* Sidebar nav buttons */
    .nav-btn {
        background: transparent !important;
        border: none !important;
        border-radius: 0 20px 20px 0 !important;
        color: #b3b3b3 !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 10px 20px 10px 24px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        margin-right: 12px !important;
        box-shadow: none !important;
        width: calc(100% - 12px) !important;
    }
    .nav-btn:hover {
        background: #333 !important;
    }
    .nav-active {
        background: #3d3d3d !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* === Import area === */
    .import-area {
        border: 2px dashed #4d4d4d !important;
        border-radius: 8px !important;
        background: #252525 !important;
    }

    /* === Right control panel === */
    .ctrl-panel {
        background: #2d2d2d !important;
        border: none !important;
        border-left: 1px solid #3d3d3d !important;
        border-radius: 0 !important;
        padding: 16px !important;
    }
    .ctrl-panel .block { background: transparent !important; border: none !important; box-shadow: none !important; }

    /* === Labels === */
    .block label span, .wrap label span {
        color: #666 !important; font-size: 10px !important; font-weight: 400 !important;
    }
    .ctrl-panel .block .info,
    .ctrl-panel .block span[data-testid="block-info"] {
        color: #e0e0e0 !important; font-size: 12px !important; font-weight: 500 !important; display: block !important;
    }

    /* === Slider === */
    input[type=range] { accent-color: #b3b3b3 !important; }

    /* === Image === */
    .image-area { background: #1d1d1d !important; border: none !important; }
    .image-area img { border-radius: 2px !important; }

    /* === Inputs === */
    textarea, input[type="text"] {
        background: #3d3d3d !important; border: 1px solid #4d4d4d !important;
        border-radius: 3px !important; color: #e0e0e0 !important; font-size: 12px !important;
    }

    /* === Buttons === */
    .process-btn {
        background: #4a90d9 !important; border: none !important; border-radius: 3px !important;
        color: #fff !important; font-size: 12px !important; font-weight: 500 !important;
        padding: 10px 20px !important; box-shadow: none !important;
    }
    .process-btn:hover { background: #5a9ee6 !important; }
    .export-btn {
        background: #2d7d46 !important; border: none !important; border-radius: 3px !important;
        color: #fff !important; font-size: 12px !important; font-weight: 500 !important;
        padding: 10px 20px !important; box-shadow: none !important; margin-top: 6px !important;
    }
    .export-btn:hover { background: #359952 !important; }

    /* === Status === */
    .status-box textarea {
        background: #252525 !important; border: 1px solid #3d3d3d !important;
        color: #8bc34a !important; font-family: 'Consolas', monospace !important;
        font-size: 11px !important; border-radius: 2px !important;
    }

    /* === Controls === */
    .block input[type="checkbox"] { accent-color: #4a90d9 !important; }
    .block select, .wrap select, .block .wrap input, .block .secondary-wrap input {
        background: #3d3d3d !important; border: 1px solid #4d4d4d !important;
        color: #e8e8e8 !important; border-radius: 3px !important; font-size: 12px !important;
    }
    .block input[type="number"] {
        background: #3d3d3d !important; border: 1px solid #4d4d4d !important;
        color: #e0e0e0 !important; font-size: 11px !important; border-radius: 2px !important; width: 45px !important;
    }

    /* === Scrollbar === */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #2d2d2d; }
    ::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f0f4ff", c100="#dbe4ff", c200="#bac8ff",
            c300="#91a7ff", c400="#748ffc", c500="#4a90d9",
            c600="#4280c4", c700="#3b6ea8", c800="#2c5282",
            c900="#1a365d", c950="#0f2440",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafafa", c100="#f5f5f5", c200="#e5e5e5",
            c300="#d4d4d4", c400="#a3a3a3", c500="#737373",
            c600="#525252", c700="#404040", c800="#2d2d2d",
            c900="#1d1d1d", c950="#141414",
        ),
    ).set(
        body_background_fill="#1d1d1d",
        body_background_fill_dark="#1d1d1d",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_label_text_color="#b3b3b3",
        block_label_text_color_dark="#b3b3b3",
        block_title_text_color="#d4d4d4",
        block_title_text_color_dark="#d4d4d4",
        input_background_fill="#3d3d3d",
        input_background_fill_dark="#3d3d3d",
        input_border_color="#4d4d4d",
        input_border_color_dark="#4d4d4d",
        button_primary_background_fill="#4a90d9",
        button_primary_background_fill_dark="#4a90d9",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        slider_color="#b3b3b3",
        slider_color_dark="#b3b3b3",
    )

    with gr.Blocks(title="WindowSeat Reflection Removal") as demo:
        # State
        original_state = gr.State(value=None)
        processed_state = gr.State(value=None)

        # Top bar
        gr.HTML('<div class="top-bar"><h2><span class="accent">WindowSeat</span>  |  Reflection Removal</h2></div>')

        with gr.Row(equal_height=False):
            # === LEFT SIDEBAR ===
            with gr.Column(scale=0, min_width=200, elem_classes="sidebar"):
                nav_import = gr.Button("📁  Import Photos", elem_classes="nav-btn nav-active", variant="secondary")
                nav_process = gr.Button("✦  Reflection Removal", elem_classes="nav-btn", variant="secondary")

            # === PANEL 1: Import Photos ===
            with gr.Column(scale=5, visible=True) as panel_import:
                gr.Markdown("### Import Images")
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Drop images here or click to browse",
                    elem_classes="import-area",
                )
                import_info = gr.Textbox(
                    label="", interactive=False, show_label=False,
                    elem_classes="status-box", lines=2,
                )
                import_preview = gr.Image(
                    type="pil", label="Preview (first image)",
                    elem_classes="image-area",
                    height=400, interactive=False,
                )

            # === PANEL 2: Reflection Removal ===
            with gr.Column(scale=5, visible=False) as panel_process:
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        with gr.Row():
                            input_image = gr.Image(
                                type="pil", label="Original",
                                elem_classes="image-area",
                                height=520, interactive=False,
                            )
                            output_image = gr.Image(
                                type="pil", label="Result",
                                elem_classes="image-area",
                                height=520, interactive=False,
                            )
                        single_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            elem_classes="status-box", lines=2,
                        )
                    # Right panel — controls
                    with gr.Column(scale=1, min_width=240, elem_classes="ctrl-panel"):
                        quality = gr.Slider(1, 5, value=2, step=1, label="Performance / Quality", info="1 = Fast, 5 = Max detail")
                        strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Strength", info="Live preview blending")
                        use_4bit = gr.Checkbox(label="4-bit Quantization", value=True, info="Reduces VRAM")
                        output_format = gr.Dropdown(["png", "jpg", "webp"], value="png", label="Format")
                        jpg_quality = gr.Slider(50, 100, value=95, step=1, label="Compression", info="JPEG/WebP quality")
                        process_btn = gr.Button("Process", variant="primary", size="lg", elem_classes="process-btn")
                        export_btn = gr.Button("Export All", variant="secondary", size="lg", elem_classes="export-btn")

        # --- Navigation switching ---
        def show_import():
            return gr.update(visible=True), gr.update(visible=False)

        def show_process():
            return gr.update(visible=False), gr.update(visible=True)

        nav_import.click(fn=show_import, outputs=[panel_import, panel_process])
        nav_process.click(fn=show_process, outputs=[panel_import, panel_process])

        # --- Events ---
        def on_upload(file_list):
            if not file_list:
                return None, "No images.", None
            first_img = Image.open(file_list[0]).convert("RGB")
            count = len(file_list)
            names = [Path(f).name for f in file_list[:5]]
            info = f"{count} image{'s' if count > 1 else ''} ready"
            if count > 5:
                info += f" — {', '.join(names)}... +{count-5} more"
            else:
                info += f" — {', '.join(names)}"
            return first_img, info, first_img

        file_upload.change(
            fn=on_upload,
            inputs=[file_upload],
            outputs=[import_preview, import_info, input_image],
        )

        def process_first(file_list, quality, strength, use_4bit, output_format, jpg_quality, progress=gr.Progress()):
            if not file_list:
                return None, None, None, "No images imported. Go to Import Photos first."
            first_img = Image.open(file_list[0]).convert("RGB")
            return web_process_single(first_img, quality, strength, use_4bit, output_format, jpg_quality, progress)

        process_btn.click(
            fn=process_first,
            inputs=[file_upload, quality, strength, use_4bit, output_format, jpg_quality],
            outputs=[output_image, original_state, processed_state, single_status],
        )
        strength.change(
            fn=update_strength_preview,
            inputs=[strength, original_state, processed_state],
            outputs=output_image,
        )
        export_btn.click(
            fn=export_all,
            inputs=[file_upload, quality, strength, use_4bit, output_format, jpg_quality],
            outputs=single_status,
        )

    demo.launch(server_name=_args.host, server_port=_args.port, inbrowser=False, theme=theme, css=css)


# ============================================================
#  Entry Point
# ============================================================
if __name__ == "__main__":
    if WEB_MODE:
        # Web mode now uses serve.py (FastAPI + React frontend)
        import uvicorn
        from backend.api import app as api_app, configure
        configure(engine=engine, mock_mode=MOCK_MODE)
        uvicorn.run(api_app, host=_args.host, port=_args.port)
    else:
        app = App()
        app.mainloop()
