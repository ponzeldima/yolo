from __future__ import annotations

import math
import re
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "cut_frames"
PREVIEW_WIDTH = 560
PREVIEW_HEIGHT = 315
GALLERY_ITEMS = 8


class VideoFrameCutter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Frame Cutter")
        self.minsize(920, 700)
        self.geometry("1180x840")

        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.duration_seconds = 0.0
        self.source_fps = 0.0
        self.total_frames = 0
        self.preview_job: str | None = None
        self.playing = False
        self.preview_image: ImageTk.PhotoImage | None = None
        self.gallery_images: list[ImageTk.PhotoImage] = []

        self.file_name = tk.StringVar(value="Choose a video file to begin.")
        self.duration_text = tk.StringVar(value="Duration: -")
        self.video_info_text = tk.StringVar(value="Video FPS: -    Size: -")
        self.start_text = tk.StringVar(value="Start: 00:00.000")
        self.end_text = tk.StringVar(value="End: 00:00.000")
        self.range_text = tk.StringVar(value="Selected range: -")
        self.frame_count_text = tk.StringVar(value="Frames to create: -")
        self.status_text = tk.StringVar(value="Select a video to preview and cut its frames.")
        self.fps_value = tk.DoubleVar(value=1.0)
        self.start_value = tk.DoubleVar(value=0.0)
        self.end_value = tk.DoubleVar(value=0.0)
        self.seek_value = tk.DoubleVar(value=0.0)

        self._build_interface()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_interface(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        source_frame = ttk.LabelFrame(self, text="Source video", padding=12)
        source_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        source_frame.columnconfigure(1, weight=1)
        ttk.Button(source_frame, text="Choose Video", command=self.choose_video).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(source_frame, textvariable=self.file_name, wraplength=850).grid(
            row=0, column=1, sticky="w", padx=12
        )
        ttk.Label(source_frame, textvariable=self.duration_text).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(source_frame, textvariable=self.video_info_text).grid(
            row=1, column=1, sticky="w", padx=12, pady=(8, 0)
        )

        controls = ttk.LabelFrame(self, text="Cut settings", padding=12)
        controls.grid(row=1, column=0, sticky="ew", padx=14, pady=8)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Frames per second:").grid(row=0, column=0, sticky="w")
        self.fps_spinbox = ttk.Spinbox(
            controls,
            from_=0.1,
            to=240.0,
            increment=0.1,
            textvariable=self.fps_value,
            width=10,
            command=self._settings_changed,
        )
        self.fps_spinbox.grid(row=0, column=1, sticky="w", padx=(8, 20))
        self.fps_spinbox.bind("<FocusOut>", self._settings_changed)
        self.fps_spinbox.bind("<Return>", self._settings_changed)
        ttk.Label(controls, textvariable=self.frame_count_text).grid(
            row=0, column=2, sticky="w"
        )

        ttk.Label(controls, textvariable=self.start_text).grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.start_scale = ttk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self.start_value,
            command=self._start_changed,
        )
        self.start_scale.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(12, 0))

        ttk.Label(controls, textvariable=self.end_text).grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.end_scale = ttk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self.end_value,
            command=self._end_changed,
        )
        self.end_scale.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(controls, textvariable=self.range_text).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        preview_frame = ttk.LabelFrame(self, text="Video preview", padding=12)
        preview_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=8)
        preview_frame.columnconfigure(0, weight=1)
        self.video_preview = ttk.Label(
            preview_frame,
            text="Video preview will appear here.",
            anchor="center",
            width=80,
        )
        self.video_preview.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.seek_scale = ttk.Scale(
            preview_frame,
            from_=0,
            to=1,
            variable=self.seek_value,
            command=self._seek_changed,
        )
        self.seek_scale.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.play_button = ttk.Button(preview_frame, text="Play", command=self.toggle_playback)
        self.play_button.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(preview_frame, text="Move the slider to inspect a video frame.").grid(
            row=2, column=1, sticky="w", padx=10, pady=(8, 0)
        )

        gallery_frame = ttk.LabelFrame(self, text="Frames that will be created", padding=12)
        gallery_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=8)
        gallery_frame.columnconfigure(tuple(range(4)), weight=1)
        self.gallery = ttk.Frame(gallery_frame)
        self.gallery.grid(row=0, column=0, sticky="nsew")
        self.gallery_placeholder = ttk.Label(
            self.gallery,
            text="Choose a video to see representative output frames.",
            anchor="center",
        )
        self.gallery_placeholder.grid(row=0, column=0, columnspan=4, pady=25)

        actions = ttk.Frame(self, padding=(14, 8, 14, 14))
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(1, weight=1)
        self.cut_button = ttk.Button(actions, text="Cut Frames", command=self.cut_frames)
        self.cut_button.grid(row=0, column=0, sticky="w")
        ttk.Label(actions, textvariable=self.status_text, wraplength=720).grid(
            row=0, column=1, sticky="w", padx=14
        )
        ttk.Button(actions, text="Start Over", command=self.start_over).grid(row=0, column=2, sticky="e")

    def choose_video(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if selected_path:
            self._load_video(Path(selected_path))

    def _load_video(self, path: Path) -> None:
        self._release_capture()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Cannot open video", f"Could not read:\n{path}")
            return

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_fps <= 0 or total_frames <= 0:
            capture.release()
            messagebox.showerror("Invalid video", "The video does not report a usable frame rate or frame count.")
            return

        self.capture = capture
        self.video_path = path
        self.source_fps = source_fps
        self.total_frames = total_frames
        self.duration_seconds = total_frames / source_fps
        self.file_name.set(str(path))
        self.duration_text.set(f"Duration: {self._format_time(self.duration_seconds)}")
        self.video_info_text.set(f"Video FPS: {source_fps:.3f}    Size: {width} x {height}")
        self.start_scale.configure(to=self.duration_seconds)
        self.end_scale.configure(to=self.duration_seconds)
        self.seek_scale.configure(to=self.duration_seconds)
        self.start_value.set(0.0)
        self.end_value.set(self.duration_seconds)
        self.seek_value.set(0.0)
        self.status_text.set("Adjust the range or extraction FPS, then cut frames.")
        self._refresh_settings()
        self._show_frame(0.0)

    def _start_changed(self, _: str = "") -> None:
        if self.start_value.get() > self.end_value.get():
            self.end_value.set(self.start_value.get())
        self._refresh_settings()

    def _end_changed(self, _: str = "") -> None:
        if self.end_value.get() < self.start_value.get():
            self.start_value.set(self.end_value.get())
        self._refresh_settings()

    def _settings_changed(self, _: object = None) -> None:
        self._refresh_settings()

    def _refresh_settings(self) -> None:
        if not self.video_path:
            return
        start_time = self.start_value.get()
        end_time = self.end_value.get()
        self.start_text.set(f"Start: {self._format_time(start_time)}")
        self.end_text.set(f"End: {self._format_time(end_time)}")
        self.range_text.set(
            f"Selected range: {self._format_time(start_time)} to {self._format_time(end_time)} "
            f"({end_time - start_time:.3f} seconds)"
        )
        try:
            frame_count = len(self._output_frame_indices())
            self.frame_count_text.set(f"Frames to create: {frame_count:,}")
        except ValueError as error:
            self.frame_count_text.set(f"Frames to create: {error}")
            return
        self._schedule_gallery_refresh()

    def _validated_output_fps(self) -> float:
        try:
            output_fps = float(self.fps_value.get())
        except tk.TclError as error:
            raise ValueError("Enter a valid extraction FPS.") from error
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("Extraction FPS must be greater than zero.")
        if self.source_fps and output_fps > self.source_fps:
            raise ValueError(f"Extraction FPS cannot exceed the video FPS ({self.source_fps:.3f}).")
        return output_fps

    def _output_frame_indices(self) -> list[int]:
        if not self.video_path:
            return []
        output_fps = self._validated_output_fps()
        start_time = self.start_value.get()
        end_time = self.end_value.get()
        sample_count = int(math.floor((end_time - start_time) * output_fps + 1e-9)) + 1
        indices: list[int] = []
        previous_index = -1
        for sample_number in range(sample_count):
            timestamp = start_time + sample_number / output_fps
            frame_index = min(int(round(timestamp * self.source_fps)), self.total_frames - 1)
            if frame_index != previous_index:
                indices.append(frame_index)
                previous_index = frame_index
        return indices

    def _seek_changed(self, _: str = "") -> None:
        if self.preview_job:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(80, lambda: self._show_frame(self.seek_value.get()))

    def _show_frame(self, timestamp: float) -> None:
        self.preview_job = None
        if not self.capture or not self.video_path:
            return
        frame_index = min(int(round(timestamp * self.source_fps)), self.total_frames - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self.capture.read()
        if not success:
            return
        image = self._frame_to_image(frame, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.preview_image = ImageTk.PhotoImage(image)
        self.video_preview.configure(image=self.preview_image, text="")

    def toggle_playback(self) -> None:
        if not self.video_path:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_next_frame()

    def _play_next_frame(self) -> None:
        if not self.playing or not self.video_path:
            return
        next_time = self.seek_value.get() + 1 / self.source_fps
        if next_time > self.duration_seconds:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.seek_value.set(next_time)
        self._show_frame(next_time)
        delay = max(1, int(1000 / min(self.source_fps, 30)))
        self.after(delay, self._play_next_frame)

    def _schedule_gallery_refresh(self) -> None:
        if self.preview_job:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(200, self._refresh_gallery)

    def _refresh_gallery(self) -> None:
        self.preview_job = None
        if not self.capture:
            return
        try:
            output_indices = self._output_frame_indices()
        except ValueError:
            return
        for widget in self.gallery.winfo_children():
            widget.destroy()
        self.gallery_images = []
        if not output_indices:
            ttk.Label(self.gallery, text="No output frames selected.").grid(row=0, column=0, pady=20)
            return
        gallery_indices = [
            output_indices[round(position * (len(output_indices) - 1) / (min(GALLERY_ITEMS, len(output_indices)) - 1))]
            if min(GALLERY_ITEMS, len(output_indices)) > 1
            else output_indices[0]
            for position in range(min(GALLERY_ITEMS, len(output_indices)))
        ]
        for position, frame_index in enumerate(gallery_indices):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = self.capture.read()
            if not success:
                continue
            image = ImageTk.PhotoImage(self._frame_to_image(frame, 230, 130))
            self.gallery_images.append(image)
            cell = ttk.Frame(self.gallery)
            cell.grid(row=position // 4, column=position % 4, padx=5, pady=5, sticky="nsew")
            ttk.Label(cell, image=image).pack()
            ttk.Label(
                cell, text=f"{self._format_time(frame_index / self.source_fps)}"
            ).pack(pady=(3, 0))

    def cut_frames(self) -> None:
        if not self.video_path:
            messagebox.showinfo("Select a video", "Choose a video file before cutting frames.")
            return
        try:
            frame_indices = self._output_frame_indices()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        if not frame_indices:
            messagebox.showerror("No frames", "The selected range does not contain any output frames.")
            return
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.video_path.stem).strip("._") or "video"
        output_directory = OUTPUT_ROOT / f"{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}"
        self.cut_button.state(["disabled"])
        self.status_text.set(f"Cutting {len(frame_indices):,} frames...")
        threading.Thread(
            target=self._write_frames,
            args=(self.video_path, frame_indices, output_directory),
            daemon=True,
        ).start()

    def _write_frames(self, video_path: Path, frame_indices: list[int], output_directory: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        written = 0
        try:
            output_directory.mkdir(parents=True, exist_ok=False)
            for sequence, frame_index in enumerate(frame_indices, start=1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                image_path = output_directory / f"frame_{sequence:06d}_source_{frame_index:08d}.jpg"
                if cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    written += 1
                if sequence % 10 == 0 or sequence == len(frame_indices):
                    self.after(0, self._set_progress, sequence, len(frame_indices))
            self.after(0, self._cut_finished, written, output_directory)
        except Exception as error:
            self.after(0, self._cut_failed, str(error))
        finally:
            capture.release()

    def _set_progress(self, completed: int, total: int) -> None:
        self.status_text.set(f"Cutting frames: {completed:,} of {total:,}")

    def _cut_finished(self, written: int, output_directory: Path) -> None:
        self.cut_button.state(["!disabled"])
        self.status_text.set(f"Created {written:,} images in {output_directory}")
        messagebox.showinfo("Frames created", f"Created {written:,} images in:\n{output_directory}")

    def _cut_failed(self, error: str) -> None:
        self.cut_button.state(["!disabled"])
        self.status_text.set("Could not cut frames.")
        messagebox.showerror("Cut failed", error)

    def start_over(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.preview_job:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        self._release_capture()
        self.video_path = None
        self.duration_seconds = 0.0
        self.source_fps = 0.0
        self.total_frames = 0
        self.file_name.set("Choose a video file to begin.")
        self.duration_text.set("Duration: -")
        self.video_info_text.set("Video FPS: -    Size: -")
        self.start_text.set("Start: 00:00.000")
        self.end_text.set("End: 00:00.000")
        self.range_text.set("Selected range: -")
        self.frame_count_text.set("Frames to create: -")
        self.status_text.set("Select a video to preview and cut its frames.")
        self.fps_value.set(1.0)
        self.start_value.set(0.0)
        self.end_value.set(0.0)
        self.seek_value.set(0.0)
        self.video_preview.configure(image="", text="Video preview will appear here.")
        self.preview_image = None
        self.gallery_images = []
        for widget in self.gallery.winfo_children():
            widget.destroy()
        self.gallery_placeholder = ttk.Label(
            self.gallery, text="Choose a video to see representative output frames.", anchor="center"
        )
        self.gallery_placeholder.grid(row=0, column=0, columnspan=4, pady=25)

    def _release_capture(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

    def _close(self) -> None:
        self._release_capture()
        self.destroy()

    @staticmethod
    def _frame_to_image(frame: cv2.typing.MatLike, max_width: int, max_height: int) -> Image.Image:
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    @staticmethod
    def _format_time(seconds: float) -> str:
        milliseconds = round((seconds % 1) * 1000)
        whole_seconds = int(seconds)
        minutes, seconds_part = divmod(whole_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if milliseconds == 1000:
            seconds_part += 1
            milliseconds = 0
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
        return f"{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


if __name__ == "__main__":
    VideoFrameCutter().mainloop()