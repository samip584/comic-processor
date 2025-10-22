"""Video generation utilities using OpenCV and FFmpeg."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoGenerator:
    """Handles video composition from panels and audio."""
    
    def __init__(self, frame_rate: int = 30, frame_size: Tuple[int, int] = (1920, 1080), 
                 zoom_in_factor: float = 1.05, zoom_out_factor: float = 1.9):
        """
        Initialize video generator.
        
        Args:
            frame_rate: Video frame rate (fps)
            frame_size: Video resolution (width, height)
            zoom_in_factor: Zoom level for zoom in effect (1.0 = no zoom, 1.05 = 5% zoom)
            zoom_out_factor: Starting zoom level for zoom out effect (1.0 = no zoom, 1.08 = 8% zoom)
        """
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.zoom_in_factor = zoom_in_factor
        self.zoom_out_factor = zoom_out_factor
        self._gradient_cache = None
        self._cached_panel_ids = None
    
    def create_video(self, panels: List[Dict], audio_segments: List[Dict],
                    analysis: Dict, output_path: Path):
        """
        Create video with correct panels at correct times (optimized with parallel frame generation).
        
        Args:
            panels: List of panel dicts with 'image', 'panel_number', 'bbox'
            audio_segments: List of audio segment dicts with durations and panel mappings
            analysis: Full analysis dict with top_panels fallback
            output_path: Where to save the final video
        """
        print(f"🎬 Creating video...")
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create video frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = temp_dir / "video.mp4"
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, self.frame_rate, self.frame_size
            )
            
            # Process each audio segment
            total_panels_shown = []
            for seg_idx, audio_seg in enumerate(audio_segments, 1):
                duration = audio_seg['duration']
                show_panels = audio_seg.get('show_panels', [])
                
                print(f"\n   📍 Segment {seg_idx}: show_panels={show_panels}, duration={duration:.2f}s")
                
                # Split into chunks of max 2 panels if needed
                panel_chunks = []
                for i in range(0, len(show_panels), 2):
                    chunk = show_panels[i:i+2]
                    panel_chunks.append(chunk)
                
                # If no panels specified, use top panels
                if not panel_chunks:
                    top_panels = analysis.get('top_panels', [])[:2]
                    panel_chunks = [top_panels]
                    print(f"   ⚠️  No panels specified, using top panels: {top_panels}")
                
                # Calculate duration per chunk
                chunk_duration = duration / len(panel_chunks) if panel_chunks else duration
                
                # Generate video for each panel chunk
                for chunk_idx, chunk_panels in enumerate(panel_chunks, 1):
                    # Get panels to show
                    panels_to_show = []
                    for panel_num in chunk_panels:
                        for panel in panels:
                            if panel['panel_number'] == panel_num:
                                panels_to_show.append(panel)
                                break
                    
                    if not panels_to_show:
                        print(f"   ⚠️  Chunk {chunk_idx}: No panels found for {chunk_panels}")
                        continue
                    
                    total_panels_shown.extend(chunk_panels)
                    
                    # Generate frames for this chunk
                    segment_frames = int(chunk_duration * self.frame_rate)
                    print(f"   🎬 Chunk {chunk_idx}/{len(panel_chunks)}: Showing panels {chunk_panels} ({segment_frames} frames, {chunk_duration:.2f}s)")
                    
                    # Choose zoom: only allow zoom out for single panels, always zoom in for multiple
                    if len(panels_to_show) == 1:
                        zoom_in = random.choice([True, False])
                    else:
                        zoom_in = True  # Always zoom in when multiple panels
                    
                    zoom_factor = self.zoom_in_factor if zoom_in else self.zoom_out_factor
                    
                    # Generate frames in parallel
                    frames = self._generate_frames_parallel(
                        panels_to_show, 
                        segment_frames, 
                        zoom_factor, 
                        zoom_in
                    )
                    
                    # Write frames to video
                    for frame in frames:
                        video_writer.write(frame)
            
            print(f"\n   ✅ Total unique panels shown in video: {sorted(set(total_panels_shown))}")
            print(f"   📊 Total panels available: {len(panels)}")
            
            video_writer.release()
            print(f"   Video frames complete!")
            
            # Combine audio
            print(f"   Combining audio segments...")
            combined_audio = temp_dir / "combined_audio.wav"
            
            # Check if all segments use the same audio file (Gemini TTS case)
            audio_paths = [seg['audio_path'] for seg in audio_segments]
            unique_audio_paths = list(set(str(p) for p in audio_paths))
            
            print(f"   📊 Audio segments: {len(audio_segments)}")
            print(f"   📊 Unique audio files: {len(unique_audio_paths)}")
            
            if len(unique_audio_paths) == 1:
                # All segments use the same audio (Gemini TTS) - just copy it
                print(f"   ✓ Using single audio file (Gemini TTS): {Path(unique_audio_paths[0]).name}")
                shutil.copy2(audio_paths[0], combined_audio)
            else:
                # Different audio files per segment (IndexTTS) - concatenate them
                print(f"   ✓ Concatenating {len(unique_audio_paths)} different audio files (IndexTTS)")
                self._combine_audio(audio_paths, combined_audio)
            
            print(f"   Audio combined!")
            
            # Get audio duration
            audio_duration = self._get_audio_duration_ffprobe(combined_audio)
            print(f"   🎵 Final audio duration: {audio_duration:.2f}s")
            
            # Merge video and audio
            print(f"   Merging video and audio...")
            self._merge_video_audio(video_path, combined_audio, output_path)
            
            print(f"✅ Video created: {output_path.name}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _generate_frames_parallel(self, panels_to_show: List[Dict], 
                                  total_frames: int, zoom_factor: float, 
                                  zoom_in: bool) -> List[np.ndarray]:
        """
        Generate video frames in parallel for better performance.
        
        Args:
            panels_to_show: List of panel dicts to display
            total_frames: Total number of frames to generate
            zoom_factor: Zoom factor for effect
            zoom_in: Whether zooming in (True) or out (False)
            
        Returns:
            List of frame arrays in order
        """
        zoom_duration = 0.5  # seconds
        zoom_frames = int(zoom_duration * self.frame_rate)
        
        # Use ThreadPoolExecutor (frame creation is I/O-bound due to cv2 operations)
        max_workers = min(4, max(1, total_frames // 10))  # Adjust based on frame count
        
        frames = [None] * total_frames
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frame generation tasks
            future_to_idx = {
                executor.submit(
                    self._generate_single_frame,
                    panels_to_show,
                    frame_idx,
                    zoom_frames,
                    zoom_factor,
                    zoom_in
                ): frame_idx
                for frame_idx in range(total_frames)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    frames[idx] = future.result()
                except Exception as e:
                    print(f"  ⚠️  Error generating frame {idx}: {e}")
                    # Create blank frame as fallback
                    frames[idx] = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        return frames
    
    def _generate_single_frame(self, panels_to_show: List[Dict], 
                              frame_idx: int, zoom_frames: int,
                              zoom_factor: float, zoom_in: bool) -> np.ndarray:
        """
        Generate a single video frame with zoom effect.
        
        Args:
            panels_to_show: Panels to display
            frame_idx: Frame index
            zoom_frames: Number of frames for zoom transition
            zoom_factor: Zoom factor
            zoom_in: Whether zooming in
            
        Returns:
            Frame as numpy array
        """
        # Calculate zoom progress for this frame
        if frame_idx < zoom_frames:
            progress = frame_idx / zoom_frames
            if zoom_in:
                zoom_progress = 1.0 + (zoom_factor - 1.0) * progress
            else:
                zoom_progress = zoom_factor - (zoom_factor - 1.0) * progress
        else:
            zoom_progress = zoom_factor if zoom_in else 1.0
        
        return self._create_frame(panels_to_show, self.frame_size, zoom_progress)
    
    def _create_frame(self, panels: List[Dict], frame_size: Tuple[int, int], zoom: float = 1.0) -> np.ndarray:
        """Create video frame showing selected panels with dynamic gradient background."""
        # Create gradient background (cached for performance)
        panel_ids = tuple(id(p['image']) for p in panels if p.get('image') is not None)
        
        if panel_ids != self._cached_panel_ids:
            # Extract dominant colors from panels
            gradient_colors = self._extract_gradient_colors(panels)
            # Create and cache gradient background
            self._gradient_cache = self._create_gradient_background(frame_size, gradient_colors)
            self._cached_panel_ids = panel_ids
        
        # Copy cached gradient
        frame = self._gradient_cache.copy()
        
        if not panels:
            return frame
        
        if len(panels) == 1:
            # Single panel - center it with zoom
            panel_img = panels[0]['image']
            resized = self._resize_panel(panel_img, frame_size, 0.95 * zoom)
            self._place_center_crop(frame, resized)
            
        else:
            # Two panels maximum (enforced earlier)
            # Check if both panels are wide (horizontal aspect ratio)
            panel1_ratio = panels[0]['image'].shape[1] / panels[0]['image'].shape[0]  # width / height
            panel2_ratio = panels[1]['image'].shape[1] / panels[1]['image'].shape[0]
            
            # If both panels have aspect ratio > 2:1 (width > 2x height), stack vertically
            if panel1_ratio >= 2.0 and panel2_ratio >= 2.0:
                # Stack vertically
                half_height = frame_size[1] // 2
                for i, panel in enumerate(panels[:2]):
                    panel_img = panel['image']
                    resized = self._resize_panel(panel_img, (frame_size[0], half_height), 0.95 * zoom)
                    x_offset = (frame_size[0] - resized.shape[1]) // 2
                    y_offset = (half_height // 2 - resized.shape[0] // 2) + (i * half_height)
                    self._place_at(frame, resized, x_offset, y_offset)
            else:
                # Two panels side by side
                half_width = frame_size[0] // 2
                for i, panel in enumerate(panels[:2]):
                    panel_img = panel['image']
                    resized = self._resize_panel(panel_img, (half_width, frame_size[1]), 0.95 * zoom)
                    x_offset = (half_width // 2 - resized.shape[1] // 2) + (i * half_width)
                    y_offset = (frame_size[1] - resized.shape[0]) // 2
                    self._place_at(frame, resized, x_offset, y_offset)
        
        return frame
    
    def _extract_gradient_colors(self, panels: List[Dict]) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from panels for gradient."""
        if not panels:
            return [(40, 40, 40), (20, 20, 20)]
        
        try:
            all_colors = []
            for panel in panels:
                img = panel['image']
                if img is None or img.size == 0:
                    continue
                
                # Downsample for faster processing
                small = cv2.resize(img, (100, 100))
                pixels = small.reshape(-1, 3).astype(np.float32)
                
                # Check if image is mostly grayscale
                std_dev = np.std(pixels, axis=0)
                is_grayscale = np.all(std_dev < 15)
                
                if is_grayscale:
                    # For grayscale, use subtle gray gradients
                    avg_brightness = np.mean(pixels)
                    if avg_brightness > 127:
                        # Light image - use light gray gradient
                        return [(240, 240, 245), (220, 220, 230)]
                    else:
                        # Dark image - use dark gray gradient
                        return [(40, 40, 45), (20, 20, 25)]
                
                # For colored images, extract dominant colors
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    colors = kmeans.cluster_centers_.astype(int)
                    all_colors.extend(colors)
                except Exception as e:
                    print(f"⚠️  Warning: KMeans clustering failed: {e}")
                    pass
            
            if not all_colors:
                return [(40, 40, 40), (20, 20, 20)]
            
            # Get 2 most representative colors
            all_colors = np.array(all_colors)
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(all_colors)
                final_colors = kmeans.cluster_centers_.astype(int)
                # Darken colors for background (reduce by 40%)
                final_colors = (final_colors * 0.6).astype(int)
                return [tuple(c) for c in final_colors]
            except Exception as e:
                print(f"⚠️  Warning: Final color processing failed: {e}")
                return [(40, 40, 40), (20, 20, 20)]
        except Exception as e:
            print(f"⚠️  Warning: Gradient color extraction failed: {e}, using default")
            return [(40, 40, 40), (20, 20, 20)]
    
    def _create_gradient_background(self, frame_size: Tuple[int, int], 
                                   colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """Create a gradient background from top to bottom."""
        if len(colors) < 2:
            colors = [(40, 40, 40), (20, 20, 20)]
        
        width, height = frame_size
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        color1 = np.array(colors[0], dtype=np.float32)
        color2 = np.array(colors[1], dtype=np.float32)
        
        # Create gradient more efficiently - only calculate per row, then broadcast
        for y in range(height):
            # Simple vertical gradient (much faster than diagonal per-pixel)
            t = y / height
            color = color1 * (1 - t) + color2 * t
            gradient[y, :] = color.astype(np.uint8)
        
        return gradient
    

    
    def _resize_panel(self, panel_img: np.ndarray, max_size: Tuple[int, int],
                     scale: float) -> np.ndarray:
        """Resize panel to fit in frame."""
        if panel_img is None or panel_img.size == 0:
            return None
        
        h, w = panel_img.shape[:2]
        max_w, max_h = int(max_size[0] * scale), int(max_size[1] * scale)
        
        scale_factor = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        return cv2.resize(panel_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def _place_center(self, frame: np.ndarray, panel: np.ndarray):
        """Place panel in center of frame."""
        if panel is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        panel_h, panel_w = panel.shape[:2]
        
        # Ensure panel fits in frame
        if panel_h > frame_h or panel_w > frame_w:
            print(f"⚠️  Warning: Panel ({panel_h}x{panel_w}) larger than frame ({frame_h}x{frame_w}), resizing...")
            scale = min(frame_h / panel_h, frame_w / panel_w) * 0.95
            new_h, new_w = int(panel_h * scale), int(panel_w * scale)
            panel = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            panel_h, panel_w = panel.shape[:2]
        
        x = (frame_w - panel_w) // 2
        y = (frame_h - panel_h) // 2
        
        # Safely place panel
        x = max(0, min(x, frame_w - panel_w))
        y = max(0, min(y, frame_h - panel_h))
        
        frame[y:y+panel_h, x:x+panel_w] = panel
    
    def _place_center_crop(self, frame: np.ndarray, panel: np.ndarray):
        """Place panel in center of frame, cropping if larger (for zoom effect)."""
        if panel is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        panel_h, panel_w = panel.shape[:2]
        
        if panel_h > frame_h or panel_w > frame_w:
            # Panel is larger - crop the center portion to fit the frame
            crop_h = min(panel_h, frame_h)
            crop_w = min(panel_w, frame_w)
            
            # Calculate crop coordinates (center of panel)
            start_y = (panel_h - crop_h) // 2
            start_x = (panel_w - crop_w) // 2
            
            # Crop the panel
            cropped_panel = panel[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            # Place cropped panel in frame (centered)
            frame_start_y = (frame_h - crop_h) // 2
            frame_start_x = (frame_w - crop_w) // 2
            
            frame[frame_start_y:frame_start_y+crop_h, frame_start_x:frame_start_x+crop_w] = cropped_panel
        else:
            # Panel fits - center it normally
            x = (frame_w - panel_w) // 2
            y = (frame_h - panel_h) // 2
            frame[y:y+panel_h, x:x+panel_w] = panel
    
    def _place_at(self, frame: np.ndarray, panel: np.ndarray, x: int, y: int):
        """Place panel at specific coordinates."""
        if panel is None:
            return
        
        panel_h, panel_w = panel.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Ensure coordinates are valid
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        
        # Clip panel if it exceeds frame boundaries
        max_panel_h = min(panel_h, frame_h - y)
        max_panel_w = min(panel_w, frame_w - x)
        
        if max_panel_h <= 0 or max_panel_w <= 0:
            return
        
        # Place only the portion that fits
        frame[y:y+max_panel_h, x:x+max_panel_w] = panel[:max_panel_h, :max_panel_w]
    
    def _combine_audio(self, audio_paths: List[Path], output_path: Path):
        """Combine multiple audio files into one with natural pauses between sentences."""
        if len(audio_paths) == 1:
            shutil.copy2(audio_paths[0], output_path)
            return
        
        # Add a 0.3 second pause between audio segments for more natural pacing
        pause_duration = 0.3  # seconds
        
        # Create temp directory for audio with pauses
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Step 1: Add silence padding to each audio file (except last)
            padded_files = []
            
            for i, audio_path in enumerate(audio_paths):
                if i < len(audio_paths) - 1:
                    # Add silence after this segment
                    padded_file = temp_dir / f"padded_{i}.wav"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(audio_path),
                        "-af", f"apad=pad_dur={pause_duration}",
                        str(padded_file)
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)
                    padded_files.append(padded_file)
                else:
                    # Last segment - no pause needed
                    padded_files.append(audio_path)
            
            # Step 2: Concatenate all padded files
            inputs = []
            for path in padded_files:
                inputs.extend(["-i", str(path)])
            
            filter_parts = [f"[{i}:0]" for i in range(len(padded_files))]
            filter_complex = f"{''.join(filter_parts)}concat=n={len(padded_files)}:v=0:a=1[out]"
            
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  Audio combining failed: {result.stderr[:300]}")
                raise Exception("FFmpeg concat failed")
                
        except Exception as e:
            print(f"⚠️  Error with pause insertion, using simple concat: {e}")
            # Fallback to simple concat without pauses
            inputs = []
            for path in audio_paths:
                inputs.extend(["-i", str(path)])
            
            filter_parts = [f"[{i}:0]" for i in range(len(audio_paths))]
            filter_complex = f"{''.join(filter_parts)}concat=n={len(audio_paths)}:v=0:a=1[out]"
            
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_audio_duration_ffprobe(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _merge_video_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Merge video and audio into final output."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
    
    @staticmethod
    def combine_videos(video_files: List[Path], output_path: Path) -> bool:
        """
        Combine multiple videos into one using FFmpeg concat.
        
        Args:
            video_files: List of video file paths to combine
            output_path: Output path for combined video
            
        Returns:
            True if successful, False otherwise
        """
        if not video_files:
            print("⚠️  No videos to combine")
            return False
        
        if len(video_files) == 1:
            print(f"ℹ️  Only one video, copying as complete video")
            shutil.copy2(video_files[0], output_path)
            print(f"✅ Video created: {output_path.name}")
            return True
        
        print(f"📹 Combining {len(video_files)} videos...")
        
        # Create temporary file list for ffmpeg
        temp_dir = Path(tempfile.mkdtemp())
        list_file = temp_dir / "concat_list.txt"
        
        try:
            # Write video list
            with open(list_file, 'w') as f:
                for video_file in video_files:
                    f.write(f"file '{video_file.absolute()}'\n")
            
            # Combine videos using ffmpeg concat
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",  # Copy without re-encoding (fast)
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Combined video created: {output_path.name}")
                
                # Calculate total duration
                total_duration = 0
                for video_file in video_files:
                    try:
                        duration_cmd = ["ffprobe", "-v", "error", "-show_entries",
                                      "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1",
                                      str(video_file)]
                        result = subprocess.run(duration_cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            total_duration += float(result.stdout.strip())
                    except:
                        pass
                
                if total_duration > 0:
                    minutes = int(total_duration // 60)
                    seconds = int(total_duration % 60)
                    print(f"   Total duration: {minutes}m {seconds}s")
                
                return True
            else:
                print(f"⚠️  Video combining failed: {result.stderr}")
                return False
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
