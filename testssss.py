import os
import cv2
import librosa
import numpy as np
from moviepy.editor import (VideoFileClip, CompositeVideoClip, ImageClip, VideoClip, 
                            concatenate_videoclips, AudioFileClip, AudioClip, ColorClip)
from sklearn.preprocessing import StandardScaler
import whisper
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import pickle
from typing import List, Tuple, Dict, Optional, Any
import warnings
from PIL import Image, ImageDraw, ImageFont
import re
from dataclasses import dataclass, field
import time
import torch
import gc
from scipy.signal import find_peaks
from collections import Counter, defaultdict
import json
from datetime import datetime, timedelta
from transformers import pipeline
import soundfile as sf
import random

warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION & MODEL SETUP
# ============================================================================

import torch.serialization
_original_torch_load = torch.load

def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass

print("\n" + "="*80)
print("🔧 GPU CONFIGURATION CHECK")
print("="*80)

CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA Available: {CUDA_AVAILABLE}")

if CUDA_AVAILABLE:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
else:
    print("⚠️ WARNING: CUDA not available! Using CPU")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

print(f"🖥️  Using Device: {DEVICE.upper()}")
print("="*80 + "\n")

# Model imports
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
    print("✓ faster-whisper available (GPU optimized)")
except ImportError as e:
    USE_FASTER_WHISPER = False
    print(f"⚠️ faster-whisper import failed: {e}")

try:
    from transformers import pipeline, CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
    USE_TRANSFORMERS = True
    print("✓ Transformers available")
except ImportError:
    USE_TRANSFORMERS = False
    print("⚠️ Install transformers: pip install transformers")

try:
    from ultralytics import YOLO
    USE_YOLO = True
    print("✓ YOLO available")
except ImportError:
    USE_YOLO = False
    print("⚠️ Install ultralytics: pip install ultralytics")


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


# ============================================================================
# VIDEO STYLE MODES
# ============================================================================

class VideoStyleMode:
    """Different video background/layout styles"""
    
    STYLES = {
        'blurred': {
            'name': 'Blurred Background (Original)',
            'description': 'Video centered with blurred background'
        },
        'black': {
            'name': 'Black Background',
            'description': 'Video centered on solid black'
        },
        'white': {
            'name': 'White Background',
            'description': 'Video centered on solid white'
        },
        'meme': {
            'name': 'Meme Style',
            'description': 'Black background, captions on top, video centered'
        },
        'split': {
            'name': 'Split Screen',
            'description': 'Video on top, black bottom for captions'
        }
    }
    
    @staticmethod
    def create_background(style: str, video_clip: VideoClip, target_size: Tuple[int, int]) -> VideoClip:
        """Create background based on style"""
        target_w, target_h = target_size
        
        if style == 'blurred':
            orig_w, orig_h = video_clip.size
            blur_scale = max(target_w / orig_w, target_h / orig_h)
            video_background = video_clip.resize(blur_scale)
            
            def make_blurred_bg(get_frame: Any, t: float) -> np.ndarray:
                frame = get_frame(t)
                bg_h, bg_w = frame.shape[:2]
                x_crop = (bg_w - target_w) // 2
                y_crop = (bg_h - target_h) // 2
                cropped = frame[y_crop:y_crop + target_h, x_crop:x_crop + target_w]
                blurred = cv2.GaussianBlur(cropped, (51, 51), 30)
                blurred = (blurred * 0.35).astype(np.uint8)
                return blurred
            
            return VideoClip(
                make_frame=lambda t: make_blurred_bg(video_background.get_frame, t),
                duration=video_clip.duration
            ).set_fps(30)
        
        elif style == 'black':
            return ColorClip(size=target_size, color=(0, 0, 0), duration=video_clip.duration)
        
        elif style == 'white':
            return ColorClip(size=target_size, color=(255, 255, 255), duration=video_clip.duration)
        
        elif style == 'meme':
            return ColorClip(size=target_size, color=(0, 0, 0), duration=video_clip.duration)
        
        elif style == 'split':
            return ColorClip(size=target_size, color=(0, 0, 0), duration=video_clip.duration)
        
        else:
            return ColorClip(size=target_size, color=(0, 0, 0), duration=video_clip.duration)


# ============================================================================
# WATERMARK SYSTEM
# ============================================================================

class WatermarkManager:
    """Add channel watermarks to video"""
    
    def __init__(self, channel_name: str = ""):
        self.channel_name = channel_name
    
    def create_watermark_clips(self, duration: float, video_size: Tuple[int, int]) -> List[ImageClip]:
        """Create watermark clips for top-right and bottom-left"""
        if not self.channel_name:
            return []
        
        watermarks = []
        font_size = 40
        font = self._load_font(font_size)
        
        # Top-right watermark
        text = f"@{self.channel_name}"
        img = Image.new('RGBA', (500, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw with outline for visibility
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((10 + dx, 10 + dy), text, font=font, fill=(0, 0, 0, 200))
        
        draw.text((10, 10), text, font=font, fill=(255, 255, 255, 220))
        
        watermark_top = ImageClip(np.array(img), duration=duration, transparent=True)
        watermark_top = watermark_top.set_position(('right', 'top')).set_start(0)
        watermarks.append(watermark_top)
        
        # Bottom-left watermark (smaller)
        font_size_small = 30
        font_small = self._load_font(font_size_small)
        img_small = Image.new('RGBA', (400, 80), (0, 0, 0, 0))
        draw_small = ImageDraw.Draw(img_small)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw_small.text((10 + dx, 10 + dy), text, font=font_small, fill=(0, 0, 0, 180))
        
        draw_small.text((10, 10), text, font=font_small, fill=(255, 255, 255, 200))
        
        watermark_bottom = ImageClip(np.array(img_small), duration=duration, transparent=True)
        watermark_bottom = watermark_bottom.set_position(('left', 'bottom')).set_start(0)
        watermarks.append(watermark_bottom)
        
        return watermarks
    
    def _load_font(self, size: int) -> Any:
        font_paths = [
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\impact.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        
        return ImageFont.load_default()


# ============================================================================
# SEGMENT SHUFFLER FOR COPYRIGHT EVASION
# ============================================================================

class SegmentShuffler:
    """Advanced segment shuffling for copyright evasion"""
    
    def __init__(self, min_chunk_size: float = 1.5, max_chunk_size: float = 3.5):
        self.min_chunk = min_chunk_size
        self.max_chunk = max_chunk_size
    
    def shuffle_segments(self, video_clip: VideoClip, audio_path: str, 
                         preserve_speech: bool = True) -> Tuple[VideoClip, str]:
        """Split into chunks and shuffle adjacent segments"""
        print("🔀 Applying segment jitter & shuffle...")
        
        duration = video_clip.duration
        chunks = []
        current_time = 0.0
        
        # Split into random-sized chunks
        while current_time < duration:
            chunk_size = random.uniform(self.min_chunk, self.max_chunk)
            chunk_end = min(current_time + chunk_size, duration)
            
            if chunk_end - current_time >= 1.0:
                chunks.append((current_time, chunk_end))
            
            current_time = chunk_end
        
        print(f"  ✓ Split into {len(chunks)} chunks")
        
        # Shuffle adjacent pairs
        shuffled_chunks = []
        i = 0
        while i < len(chunks):
            if i + 1 < len(chunks) and random.random() < 0.6:
                shuffled_chunks.append(chunks[i + 1])
                shuffled_chunks.append(chunks[i])
                i += 2
            else:
                shuffled_chunks.append(chunks[i])
                i += 1
        
        # Create video clips
        video_segments = []
        for start, end in shuffled_chunks:
            try:
                segment = video_clip.subclip(start, end)
                video_segments.append(segment)
            except Exception as e:
                print(f"  ⚠️ Skipping chunk {start:.1f}-{end:.1f}: {e}")
        
        if not video_segments:
            print("  ⚠️ No segments to shuffle, returning original")
            return video_clip, audio_path
        
        # Concatenate shuffled segments
        try:
            final_video = concatenate_videoclips(video_segments, method="compose")
        except Exception as e:
            print(f"  ⚠️ Shuffle failed: {e}")
            return video_clip, audio_path
        
        # Shuffle audio similarly
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
            audio_segments = []
            
            for start, end in shuffled_chunks:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                if end_sample <= len(y):
                    audio_segments.append(y[start_sample:end_sample])
            
            shuffled_audio = np.concatenate(audio_segments) if audio_segments else y
            
            shuffled_audio_path = "temp_shuffled_audio.wav"
            sf.write(shuffled_audio_path, shuffled_audio, sr)
            
            print(f"  ✓ Shuffled {len(shuffled_chunks)} segments")
            return final_video, shuffled_audio_path
            
        except Exception as e:
            print(f"  ⚠️ Audio shuffle failed: {e}")
            return final_video, audio_path


# ============================================================================
# SMOOTH RANDOM ZOOM
# ============================================================================

class SmoothZoomEffect:
    """Apply smooth random zooms throughout video"""
    
    def __init__(self, zoom_probability: float = 0.3, max_zoom: float = 1.15, base_zoom: float = 1.3):
        self.zoom_prob = zoom_probability
        self.max_zoom = max_zoom
        self.base_zoom = base_zoom  # Permanent zoom level
    
    def apply_zoom(self, video_clip: VideoClip) -> VideoClip:
        """Apply permanent base zoom + smooth random zooms on top"""
        print(f"🔍 Applying permanent {self.base_zoom}x zoom + smooth random zooms...")
        
        duration = video_clip.duration
        fps = video_clip.fps
        total_frames = int(duration * fps)
        
        # Generate zoom timeline (on top of base zoom)
        zoom_timeline = []
        current_zoom = 1.0  # This is relative to base_zoom
        target_zoom = 1.0
        
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            
            if random.random() < self.zoom_prob / fps:
                target_zoom = random.uniform(1.0, self.max_zoom)
            
            current_zoom += (target_zoom - current_zoom) * 0.02
            zoom_timeline.append(current_zoom)
        
        def zoom_frame(get_frame, t):
            frame = get_frame(t)
            frame_idx = int(t * fps)
            
            if frame_idx >= len(zoom_timeline):
                zoom_multiplier = 1.0
            else:
                zoom_multiplier = zoom_timeline[frame_idx]
            
            # Apply base zoom + dynamic zoom
            total_zoom = self.base_zoom * zoom_multiplier
            
            h, w = frame.shape[:2]
            new_h, new_w = int(h / total_zoom), int(w / total_zoom)
            
            # Crop from center
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            
            cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return zoomed
        
        zoomed_clip = video_clip.fl(lambda gf, t: zoom_frame(gf, t))
        
        avg_zoom = np.mean(zoom_timeline) * self.base_zoom
        max_zoom_applied = np.max(zoom_timeline) * self.base_zoom
        print(f"  ✓ Base zoom: {self.base_zoom}x | Dynamic range: {self.base_zoom:.2f}x-{max_zoom_applied:.2f}x")
        
        return zoomed_clip


# ============================================================================
# BACKGROUND MUSIC MIXER
# ============================================================================

class BackgroundMusicMixer:
    """Mix background music with original audio"""
    
    def __init__(self, music_path: Optional[str] = None, music_volume: float = 0.3):
        self.music_path = music_path
        self.music_volume = music_volume
    
    def mix_audio(self, original_audio_path: str, duration: float, 
                  speech_timestamps: List[Tuple[float, float]] = None) -> str:
        """Mix background music with original audio (ducking under speech)"""
        
        if not self.music_path or not os.path.exists(self.music_path):
            print("  ℹ️ No background music specified")
            return original_audio_path
        
        print("🎵 Mixing background music...")
        
        try:
            original, sr = librosa.load(original_audio_path, sr=44100, mono=True)
            music, music_sr = librosa.load(self.music_path, sr=44100, mono=True)
            
            target_samples = int(duration * sr)
            if len(music) < target_samples:
                repeats = int(np.ceil(target_samples / len(music)))
                music = np.tile(music, repeats)
            
            music = music[:target_samples]
            
            if len(original) < target_samples:
                original = np.pad(original, (0, target_samples - len(original)))
            else:
                original = original[:target_samples]
            
            music = music * self.music_volume
            
            # Apply ducking
            if speech_timestamps:
                ducking_envelope = np.ones(len(music))
                
                for start, end in speech_timestamps:
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    
                    if end_sample <= len(ducking_envelope):
                        ducking_envelope[start_sample:end_sample] *= 0.3
                
                from scipy.ndimage import gaussian_filter1d
                ducking_envelope = gaussian_filter1d(ducking_envelope, sigma=sr * 0.1)
                music = music * ducking_envelope
            
            mixed = original + music
            
            max_val = np.max(np.abs(mixed))
            if max_val > 0:
                mixed = mixed / max_val * 0.95
            
            mixed_path = "temp_mixed_audio.wav"
            sf.write(mixed_path, mixed, sr)
            
            print(f"  ✓ Music mixed at {self.music_volume * 100:.0f}% volume (ducked under speech)")
            
            return mixed_path
            
        except Exception as e:
            print(f"  ⚠️ Music mixing failed: {e}")
            return original_audio_path


# ============================================================================
# ADVANCED AUDIO PROCESSOR
# ============================================================================

class AdvancedAudioProcessor:
    """Advanced audio transformations for copyright evasion"""
    
    def __init__(self):
        self.pitch_range = (-4, 4)
        self.speed_range = (0.97, 1.03)
        self.stretch_range = (0.98, 1.02)
    
    def apply_advanced_transforms(self, audio_path: str) -> str:
        """Apply multiple audio transforms"""
        print("🎚️ Applying advanced audio transforms...")
        
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
            
            # 1. Pitch shift
            pitch_shift_semitones = random.uniform(*self.pitch_range)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_semitones)
            print(f"  ✓ Pitch shifted: {pitch_shift_semitones:+.2f} semitones")
            
            # 2. Time stretch
            stretch_factor = random.uniform(*self.stretch_range)
            y = librosa.effects.time_stretch(y, rate=stretch_factor)
            print(f"  ✓ Time stretched: {stretch_factor:.4f}x")
            
            # 3. Speed change
            speed_factor = random.uniform(*self.speed_range)
            y = librosa.effects.time_stretch(y, rate=speed_factor)
            print(f"  ✓ Speed adjusted: {speed_factor:.4f}x")
            
            # 4. Add subtle noise
            noise = np.random.normal(0, 0.001, len(y))
            y = y + noise
            
            # 5. Subtle EQ adjustment
            y = self._apply_subtle_eq(y, sr)
            
            # Normalize
            y = y / np.max(np.abs(y)) * 0.95
            
            output_path = "temp_transformed_audio.wav"
            sf.write(output_path, y, sr)
            
            print("  ✓ All audio transforms applied")
            return output_path
            
        except Exception as e:
            print(f"  ⚠️ Audio transform failed: {e}")
            return audio_path
    
    def _apply_subtle_eq(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply subtle EQ changes"""
        try:
            stft = librosa.stft(y)
            mag, phase = np.abs(stft), np.angle(stft)
            
            freq_bins = mag.shape[0]
            eq_curve = np.random.uniform(0.95, 1.05, freq_bins)
            
            mag = mag * eq_curve[:, np.newaxis]
            
            y_eq = librosa.istft(mag * np.exp(1j * phase))
            return y_eq
        except:
            return y


# ============================================================================
# CONTENT ID EVASION (from Document 2)
# ============================================================================

@dataclass
class ModificationSegment:
    """Segment marked for modification"""
    start_time: float
    end_time: float
    modification_type: str
    factor: float
    priority: float
    can_modify: bool


class ContentIDEvasion:
    """Smart modifications to evade Content ID"""
    
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        self.motion_threshold = 15.0 if aggressive else 20.0
        self.silence_percentile = 25 if aggressive else 20
        self.min_cut_duration = 0.15
        self.max_cut_duration = 1.5
        self.pitch_range = (0.94, 1.06)
        self.speed_range = (0.96, 1.08)
        
    def identify_cuttable_segments(self, features: Dict[str, Any], clip_duration: float) -> List[ModificationSegment]:
        """Find segments safe to modify"""
        print("✂️  Identifying modification opportunities...")
        
        motion = features['motion_scores']
        audio_energy = features['audio_energy']
        transcript_segments = features['transcript_segments']
        duration = min(features['duration'], clip_duration)
        
        speech_timeline = np.zeros(int(duration) + 1, dtype=bool)
        for seg in transcript_segments:
            start_idx = max(0, int(seg['start']))
            end_idx = min(len(speech_timeline), int(seg['end']) + 1)
            if start_idx < len(speech_timeline) and end_idx <= len(speech_timeline):
                speech_timeline[start_idx:end_idx] = True
        
        segments: List[ModificationSegment] = []
        window_size = 2
        
        for start_sec in range(0, int(duration) - window_size):
            end_sec = min(start_sec + window_size, duration)
            
            start_idx = int(start_sec * len(motion) / features['duration'])
            end_idx = int(end_sec * len(motion) / features['duration'])
            
            if end_idx >= len(motion) or start_idx >= len(motion):
                continue
            
            motion_score = float(np.mean(motion[start_idx:end_idx]))
            audio_score = float(np.mean(audio_energy[start_idx:end_idx]))
            has_speech = np.any(speech_timeline[start_sec:int(end_sec)]) if start_sec < len(speech_timeline) else False
            
            cut_priority = 0.0
            
            if motion_score < self.motion_threshold:
                cut_priority += 40
            
            if audio_score < np.percentile(audio_energy, self.silence_percentile):
                cut_priority += 30
            
            if not has_speech:
                cut_priority += 30
            else:
                cut_priority -= 20
            
            can_modify = cut_priority > 50 and not has_speech
            
            if can_modify:
                if motion_score < self.motion_threshold * 0.7 and audio_score < np.percentile(audio_energy, 15):
                    mod_type = 'cut'
                    factor = 0.0
                elif motion_score < self.motion_threshold:
                    mod_type = 'speed'
                    factor = np.random.uniform(1.05, 1.10)
                else:
                    mod_type = 'pitch'
                    factor = np.random.uniform(*self.pitch_range)
            else:
                if np.random.random() < 0.3:
                    mod_type = 'pitch'
                    factor = np.random.uniform(0.98, 1.02)
                    can_modify = True
                    cut_priority = 10
                else:
                    continue
            
            segment = ModificationSegment(
                start_time=float(start_sec),
                end_time=float(end_sec),
                modification_type=mod_type,
                factor=factor,
                priority=cut_priority,
                can_modify=can_modify
            )
            segments.append(segment)
        
        merged = self._merge_segments(segments)
        
        cut_count = sum(1 for s in merged if s.modification_type == 'cut')
        speed_count = sum(1 for s in merged if s.modification_type == 'speed')
        pitch_count = sum(1 for s in merged if s.modification_type == 'pitch')
        
        print(f"  ✓ Found {len(merged)} modification zones:")
        print(f"    • {cut_count} segments to cut")
        print(f"    • {speed_count} segments to speed up")
        print(f"    • {pitch_count} segments to pitch shift")
        
        return merged
    
    def _merge_segments(self, segments: List[ModificationSegment]) -> List[ModificationSegment]:
        """Merge adjacent similar segments"""
        if not segments:
            return []
        
        segments.sort(key=lambda x: x.start_time)
        merged = [segments[0]]
        
        for seg in segments[1:]:
            prev = merged[-1]
            
            if (seg.modification_type == prev.modification_type and
                seg.start_time - prev.end_time < 1.0 and
                abs(seg.factor - prev.factor) < 0.05):
                
                merged[-1] = ModificationSegment(
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    modification_type=prev.modification_type,
                    factor=(prev.factor + seg.factor) / 2,
                    priority=(prev.priority + seg.priority) / 2,
                    can_modify=True
                )
            else:
                merged.append(seg)
        
        return merged
    
    def apply_smart_modifications(self, video_clip: VideoClip, segments: List[ModificationSegment],
                                  audio_path: str) -> Tuple[VideoClip, str]:
        """Apply modifications efficiently"""
        print("🎬 Applying smart modifications...")
        
        clip_duration = video_clip.duration
        
        valid_segments = []
        for seg in segments:
            if seg.start_time >= clip_duration:
                continue
            if seg.end_time > clip_duration:
                seg.end_time = clip_duration
            if seg.end_time - seg.start_time >= 0.5:
                valid_segments.append(seg)
        
        segments = valid_segments
        
        if not segments:
            print("  ℹ️ No valid modification segments found")
            return video_clip, audio_path
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
        except Exception as e:
            print(f"  ⚠️ Could not load audio: {e}")
            return video_clip, audio_path
        
        segments.sort(key=lambda x: x.start_time)
        
        clips = []
        audio_segments = []
        current_time = 0.0
        
        for seg in segments:
            seg.start_time = max(0.0, min(seg.start_time, clip_duration))
            seg.end_time = max(seg.start_time, min(seg.end_time, clip_duration))
            
            if current_time < seg.start_time and seg.start_time <= clip_duration:
                try:
                    normal_clip = video_clip.subclip(current_time, min(seg.start_time, clip_duration))
                    clips.append(normal_clip)
                    
                    start_sample = int(current_time * sr)
                    end_sample = int(min(seg.start_time, clip_duration) * sr)
                    if start_sample < len(y) and end_sample <= len(y):
                        audio_segments.append(y[start_sample:end_sample])
                except Exception as e:
                    print(f"  ⚠️ Skipping segment {current_time}-{seg.start_time}: {e}")
            
            if seg.modification_type == 'cut':
                pass
            
            elif seg.modification_type == 'speed':
                try:
                    if seg.start_time < clip_duration and seg.end_time <= clip_duration:
                        segment = video_clip.subclip(seg.start_time, seg.end_time)
                        speeded = segment.speedx(seg.factor)
                        clips.append(speeded)
                        
                        start_sample = int(seg.start_time * sr)
                        end_sample = int(seg.end_time * sr)
                        if start_sample < len(y) and end_sample <= len(y):
                            audio_seg = y[start_sample:end_sample]
                            
                            new_length = int(len(audio_seg) / seg.factor)
                            if new_length > 0:
                                audio_seg_stretched = librosa.effects.time_stretch(audio_seg, rate=seg.factor)
                                audio_segments.append(audio_seg_stretched)
                except Exception as e:
                    print(f"  ⚠️ Skipping speed modification at {seg.start_time}: {e}")
            
            elif seg.modification_type == 'pitch':
                try:
                    if seg.start_time < clip_duration and seg.end_time <= clip_duration:
                        segment = video_clip.subclip(seg.start_time, seg.end_time)
                        clips.append(segment)
                        
                        start_sample = int(seg.start_time * sr)
                        end_sample = int(seg.end_time * sr)
                        if start_sample < len(y) and end_sample <= len(y):
                            audio_seg = y[start_sample:end_sample]
                            
                            semitones = 12 * np.log2(seg.factor)
                            try:
                                audio_seg_shifted = librosa.effects.pitch_shift(audio_seg, sr=sr, n_steps=semitones)
                                audio_segments.append(audio_seg_shifted)
                            except:
                                audio_segments.append(audio_seg)
                except Exception as e:
                    print(f"  ⚠️ Skipping pitch modification at {seg.start_time}: {e}")
            
            current_time = seg.end_time
        
        if current_time < clip_duration:
            try:
                final_clip = video_clip.subclip(current_time, clip_duration)
                clips.append(final_clip)
                
                start_sample = int(current_time * sr)
                if start_sample < len(y):
                    audio_segments.append(y[start_sample:])
            except Exception as e:
                print(f"  ⚠️ Could not add final segment: {e}")
        
        if not clips:
            print("  ℹ️ No clips to concatenate")
            return video_clip, audio_path
        
        try:
            print("  🔗 Concatenating video segments...")
            final_video = concatenate_videoclips(clips, method="compose")
        except Exception as e:
            print(f"  ⚠️ Concatenation failed: {e}")
            return video_clip, audio_path
        
        print("  🎵 Merging modified audio...")
        final_audio = np.concatenate(audio_segments) if audio_segments else y
        
        modified_audio_path = "temp_modified_audio.wav"
        try:
            sf.write(modified_audio_path, final_audio, sr)
        except Exception as e:
            print(f"  ⚠️ Could not save modified audio: {e}")
            return final_video, audio_path
        
        print(f"  ✓ Modifications applied!")
        print(f"    • Original duration: {video_clip.duration:.1f}s")
        print(f"    • New duration: {final_video.duration:.1f}s")
        print(f"    • Time saved: {video_clip.duration - final_video.duration:.1f}s")
        
        return final_video, modified_audio_path


# ============================================================================
# PERFORMANCE TRACKING & LEARNING
# ============================================================================

class PerformanceTracker:
    """Track and learn from historical short performance"""
    
    def __init__(self, db_path: str = 'shorts_performance.json'):
        self.db_path = db_path
        self.performance_data: List[Dict[str, Any]] = self.load_data()
        print(f"📊 Performance Tracker: Loaded {len(self.performance_data)} historical records")
    
    def load_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load performance data: {e}")
                return []
        return []
    
    def save_data(self) -> None:
        with open(self.db_path, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
    
    def log_short_creation(self, short_metadata: Dict[str, Any], video_path: str) -> str:
        """Log when a short is created"""
        record_id = f"short_{int(time.time())}_{len(self.performance_data)}"
        record = {
            'id': record_id,
            'created_at': datetime.now().isoformat(),
            'video_path': video_path,
            'clip_features': short_metadata,
            'performance': None,
            'status': 'created'
        }
        self.performance_data.append(record)
        self.save_data()
        print(f"  📝 Logged creation: {record_id}")
        return record_id
    
    def update_performance(self, record_id: str, youtube_analytics: Dict[str, Any]) -> None:
        """Update with YouTube analytics"""
        for record in self.performance_data:
            if record['id'] == record_id:
                record['performance'] = {
                    'views': youtube_analytics.get('views', 0),
                    'avg_view_duration': youtube_analytics.get('avg_duration', 0),
                    'likes': youtube_analytics.get('likes', 0),
                    'comments': youtube_analytics.get('comments', 0),
                    'shares': youtube_analytics.get('shares', 0),
                    'ctr': youtube_analytics.get('ctr', 0),
                    'retention_curve': youtube_analytics.get('retention', []),
                    'updated_at': datetime.now().isoformat()
                }
                record['status'] = 'tracked'
                self.save_data()
                print(f"  ✅ Updated analytics for: {record_id}")
                return
        print(f"  ⚠️ Record not found: {record_id}")
    
    def get_best_features(self, min_views: int = 1000) -> Dict[str, Any]:
        """Learn from top-performing shorts"""
        tracked = [r for r in self.performance_data 
                   if r['status'] == 'tracked' and r['performance']]
        
        if not tracked:
            print("  ℹ️ No tracked performance data yet. Using defaults.")
            return {}
        
        good_performers = [r for r in tracked 
                          if r['performance']['views'] >= min_views]
        
        if not good_performers:
            good_performers = sorted(tracked, 
                                   key=lambda x: x['performance']['views'], 
                                   reverse=True)[:max(1, len(tracked)//3)]
        
        print(f"  📈 Analyzing {len(good_performers)} top performers...")
        
        durations = [r['clip_features'].get('actual_duration', 60) 
                    for r in good_performers]
        content_types = [r['clip_features'].get('content_type', 'unknown') 
                        for r in good_performers]
        all_triggers = [trigger 
                       for r in good_performers 
                       for trigger in r['clip_features'].get('viral_triggers', [])]
        avg_scores = [r['clip_features'].get('viral_potential', 0) 
                     for r in good_performers]
        speech_ratios = [r['clip_features'].get('speech_ratio', 0.5) 
                        for r in good_performers]
        
        learned = {
            'avg_duration': np.mean(durations) if durations else 60,
            'duration_std': np.std(durations) if len(durations) > 1 else 10,
            'common_content_types': Counter(content_types).most_common(3),
            'successful_triggers': Counter(all_triggers).most_common(5),
            'avg_viral_score': np.mean(avg_scores) if avg_scores else 50,
            'optimal_speech_ratio': np.mean(speech_ratios) if speech_ratios else 0.6,
            'sample_size': len(good_performers)
        }
        
        print(f"  ✓ Learned patterns from {learned['sample_size']} shorts:")
        print(f"    • Optimal duration: {learned['avg_duration']:.1f}s")
        print(f"    • Top content types: {[t[0] for t in learned['common_content_types'][:2]]}")
        print(f"    • Best triggers: {[t[0] for t in learned['successful_triggers'][:3]]}")
        
        return learned


class AnalysisCache:
    """Cache video analysis"""
    
    def __init__(self, cache_dir: str = 'analysis_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"💾 Analysis Cache: {cache_dir}")
    
    def get_cache_key(self, video_path: str) -> str:
        try:
            file_stat = os.stat(video_path)
            return f"{os.path.basename(video_path)}_{file_stat.st_size}_{int(file_stat.st_mtime)}"
        except Exception:
            return f"{os.path.basename(video_path)}_nocache"
    
    def load_cached_analysis(self, video_path: str) -> Optional[Dict[str, Any]]:
        cache_key = self.get_cache_key(video_path)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"  ✓ Loaded cached analysis")
                    return data
            except Exception as e:
                print(f"  ⚠️ Cache load failed: {e}")
        return None
    
    def save_analysis(self, video_path: str, features: Dict[str, Any], analysis_time: float) -> None:
        cache_key = self.get_cache_key(video_path)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        features['analysis_time'] = analysis_time
        features['cached_at'] = datetime.now().isoformat()
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            print(f"  💾 Cached analysis for future use")
        except Exception as e:
            print(f"  ⚠️ Cache save failed: {e}")


class ContentPatterns:
    """Pre-built patterns from YouTube Shorts research"""
    
    CONTENT_TYPE_PATTERNS = {
        'tutorial': {
            'optimal_duration': (45, 75),
            'score_multiplier': 1.3
        },
        'comedy': {
            'optimal_duration': (30, 60),
            'score_multiplier': 1.2
        },
        'product review': {
            'optimal_duration': (50, 80),
            'score_multiplier': 1.4
        },
        'reaction video': {
            'optimal_duration': (35, 65),
            'score_multiplier': 1.1
        },
        'transformation': {
            'optimal_duration': (40, 70),
            'score_multiplier': 1.5
        }
    }
    
    @staticmethod
    def get_content_boost(content_type: str, duration: float, speech_ratio: float) -> float:
        pattern = ContentPatterns.CONTENT_TYPE_PATTERNS.get(content_type.lower(), {})
        
        if not pattern:
            return 1.0
        
        boost = pattern.get('score_multiplier', 1.0)
        
        opt_dur = pattern.get('optimal_duration', (30, 90))
        if opt_dur[0] <= duration <= opt_dur[1]:
            boost *= 1.1
        
        return boost


class TimeEstimator:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        
    def start(self) -> None:
        self.start_time = time.time()
        
    def get_elapsed(self) -> str:
        if self.start_time is None:
            return "0s"
        elapsed = time.time() - self.start_time
        return self.format_time(elapsed)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


@dataclass
class ContextualMoment:
    """Rich contextual information about a moment"""
    timestamp: float
    duration: float
    speech_text: str = ""
    speech_sentiment: str = ""
    speech_confidence: float = 0.0
    speech_emotion: str = ""
    entities: List[str] = field(default_factory=list)
    visual_scene: str = ""
    visual_objects: List[str] = field(default_factory=list)
    visual_actions: List[str] = field(default_factory=list)
    face_emotions: List[str] = field(default_factory=list)
    audio_intensity: float = 0.0
    audio_emotion: str = ""
    music_present: bool = False
    viral_score: float = 0.0
    viral_triggers: List[str] = field(default_factory=list)
    hook_strength: float = 0.0
    retention_score: float = 0.0
    content_type: str = ""
    narrative_role: str = ""
    standalone_quality: float = 0.0


# ============================================================================
# ADVANCED AI ANALYZER (Core Analysis Engine)
# ============================================================================

class AdvancedAIAnalyzer:
    """State-of-the-art multi-modal AI analysis system"""
    
    def __init__(self, clip_duration: int = 60, use_learning: bool = True, apply_anti_copyright: bool = True) -> None:
        self.clip_duration = clip_duration
        self.timer = TimeEstimator()
        self.use_learning = use_learning
        self.apply_anti_copyright = apply_anti_copyright
        
        if apply_anti_copyright:
            self.evasion = ContentIDEvasion(aggressive=False)
            print("🛡️  Content ID Evasion: ENABLED")
        else:
            self.evasion = None
            print("⚠️  Content ID Evasion: DISABLED")
        
        if use_learning:
            self.performance_tracker = PerformanceTracker()
            self.analysis_cache = AnalysisCache()
            self.learned_patterns = self.performance_tracker.get_best_features()
            
            if self.learned_patterns and 'avg_duration' in self.learned_patterns:
                learned_duration = int(self.learned_patterns['avg_duration'])
                if 30 <= learned_duration <= 90:
                    self.clip_duration = learned_duration
                    print(f"  🎯 Using learned optimal duration: {self.clip_duration}s")
        else:
            self.performance_tracker = None
            self.analysis_cache = None
            self.learned_patterns = {}
        
        self.whisper_model: Any = None
        self.clip_model: Any = None
        self.clip_processor: Any = None
        self.blip_model: Any = None
        self.blip_processor: Any = None
        self.sentiment_analyzer: Any = None
        self.emotion_analyzer: Any = None
        self.content_classifier: Any = None
        self.object_detector: Any = None
        self.use_yolo: bool = False
        
        print("\n🔧 Loading AI Models...")
        print("="*80)
        
        # Load Whisper
        if USE_FASTER_WHISPER:
            print("Loading Whisper model...")
            if DEVICE == "cuda":
                self.whisper_model = WhisperModel(
                    "base", 
                    device="cuda",
                    compute_type="float16",
                    device_index=0,
                    num_workers=4
                )
            else:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)
        
        # Load vision/NLP models
        if USE_TRANSFORMERS:
            print("Loading CLIP...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if DEVICE == "cuda":
                self.clip_model = self.clip_model.to("cuda")
                self.clip_model.half()
            
            print("Loading BLIP...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            if DEVICE == "cuda":
                self.blip_model = self.blip_model.to("cuda")
                self.blip_model.half()

            print("Loading sentiment analyzer...")
            device_id = 0 if torch.cuda.is_available() else -1
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device_id
            )
            
            print("Loading emotion detector...")
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=device_id
            )
            
            print("Loading content classifier...")
            self.content_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device_id
            )
        
        if USE_YOLO:
            print("Loading YOLO...")
            try:
                self.object_detector = YOLO('yolov8n.pt')
                if DEVICE == "cuda":
                    self.object_detector.to('cuda')
                self.use_yolo = True
            except Exception as e:
                print(f"    ⚠️  YOLO failed: {e}")
                self.use_yolo = False
        
        # Viral patterns
        self.viral_patterns: Dict[str, List[str]] = {
            'hooks': [
                r'\b(wait|stop|hold on|hold up|pause)\b',
                r'\b(no way|omg|wow|insane|crazy|wild|unbelievable)\b',
                r'\b(you won\'?t believe|can\'?t believe)\b',
            ],
            'questions': [
                r'\?',
                r'\b(why|how|what|when|where|who)\b',
            ],
            'revelations': [
                r'\b(secret|truth|reveal|exposed|hidden)\b',
            ],
            'intensity': [
                r'\b(best|worst|most|least|never|always)\b',
            ],
        }
        
        if self.learned_patterns and 'successful_triggers' in self.learned_patterns:
            learned_triggers = [t[0] for t in self.learned_patterns['successful_triggers']]
            if learned_triggers:
                self.viral_patterns['learned'] = [
                    r'\b' + re.escape(trigger.lower()) + r'\b' 
                    for trigger in learned_triggers[:5]
                ]
        
        self.content_types: List[str] = [
            "tutorial or how-to",
            "product review",
            "reaction video",
            "storytelling",
            "comedy",
            "transformation",
            "educational",
        ]
        
        self.action_labels: List[str] = [
            "person talking to camera",
            "person reacting",
            "showing a product",
            "demonstrating",
            "cooking",
            "dancing",
        ]
        
        print("✓ All models loaded!")
        print("="*80 + "\n")
    
    def download_video(self, url: str, output_path: str = "temp_video.mp4") -> Tuple[str, str]:
        import yt_dlp
        
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        print(f"📥 Downloading video...")
        download_start = time.time()
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video') if info else 'video'
        
        download_time = time.time() - download_start
        print(f"✓ Downloaded: {title} ({self.timer.format_time(download_time)})\n")
        return output_path, title
        
    def get_video_path(self, input_source: str) -> Tuple[str, str]:
        if os.path.exists(input_source):
            print(f"📁 Using local video: {input_source}")
            title = os.path.splitext(os.path.basename(input_source))[0]
            return input_source, title
        
        videos_folder = os.path.join(os.getcwd(), "videos")
        local_path = os.path.join(videos_folder, input_source)
        
        if os.path.exists(local_path):
            print(f"📁 Using local video from videos folder: {input_source}")
            title = os.path.splitext(input_source)[0]
            return local_path, title
        
        if not ('youtube.com' in input_source or 'youtu.be' in input_source):
            raise ValueError("Input is neither a valid file path nor a YouTube URL")
        
        return self.download_video(input_source)
    
    def analyze_visual_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        if not USE_TRANSFORMERS:
            return {'scene': 'unknown', 'actions': [], 'objects': []}
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        inputs = self.blip_processor(pil_image, return_tensors="pt")
        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        inputs = self.clip_processor(
            text=self.action_labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]
        
        top_actions: List[str] = []
        for idx in torch.topk(probs, k=3).indices:
            if probs[idx] > 0.15:
                top_actions.append(self.action_labels[int(idx)])
        
        objects: List[str] = []
        if self.use_yolo and self.object_detector:
            results = self.object_detector(frame, verbose=False)
            if len(results) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        obj_name = results[0].names[class_id]
                        objects.append(obj_name)
        
        return {
            'scene': caption,
            'actions': top_actions,
            'objects': list(set(objects))
        }
    
    def analyze_speech_deep(self, text: str) -> Dict[str, Any]:
        if not text or not USE_TRANSFORMERS:
            return {
                'sentiment': 'neutral',
                'emotion': 'neutral',
                'entities': [],
                'viral_triggers': [],
                'content_type': 'unknown'
            }
        
        sentiment_result = self.sentiment_analyzer(text[:512])[0]
        sentiment = sentiment_result['label'].lower()
        
        emotion_results = self.emotion_analyzer(text[:512])[0]
        emotion = max(emotion_results, key=lambda x: x['score'])['label']
        
        entities: List[str] = []
        prices = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        entities.extend(prices)
        
        viral_triggers: List[str] = []
        for category, patterns in self.viral_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    viral_triggers.append(category)
                    break
        
        if len(text) > 10:
            content_result = self.content_classifier(
                text[:512],
                candidate_labels=self.content_types,
                multi_label=False
            )
            content_type = content_result['labels'][0]
        else:
            content_type = 'unknown'
        
        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'entities': entities,
            'viral_triggers': list(set(viral_triggers)),
            'content_type': content_type
        }
    
    def analyze_video_contextual(self, video_path: str) -> Dict[str, Any]:
        """Deep contextual analysis with caching"""
        
        if self.analysis_cache:
            cached = self.analysis_cache.load_cached_analysis(video_path)
            if cached:
                return cached
        
        print("🧠 Performing deep contextual analysis...")
        print("=" * 80)
        analysis_start = time.time()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"  📹 Video: {duration:.1f}s @ {fps:.1f} FPS")
        
        sample_rate = max(1, int(fps * 2))
        
        motion_scores: List[float] = []
        visual_contexts: List[Dict[str, Any]] = []
        scene_changes: List[float] = []
        face_presence: List[int] = []
        prev_frame: Optional[np.ndarray] = None
        frame_idx = 0
        
        print("  📹 Analyzing visual content...")
        analyzed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                
                gray = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion_score = float(np.mean(diff))
                    motion_scores.append(motion_score)
                    
                    if motion_score > 45:
                        scene_changes.append(timestamp)
                
                prev_frame = gray
                
                if frame_idx % (sample_rate * 3) == 0 and USE_TRANSFORMERS:
                    visual_context = self.analyze_visual_scene(frame)
                    visual_contexts.append({
                        'timestamp': timestamp,
                        **visual_context
                    })
                    analyzed_frames += 1
                
                face_presence.append(0)
            
            frame_idx += 1
        
        cap.release()
        print(f"    ✓ Analyzed {analyzed_frames} frames")
        
        print("  🔊 Analyzing audio...")
        
        try:
            y, sr = librosa.load(video_path, sr=22050, duration=duration, mono=True)
            
            energy = librosa.feature.rms(y=y, hop_length=512)[0]
            energy_peaks, _ = find_peaks(energy, prominence=np.std(energy) * 0.5)
            energy_peak_times = librosa.frames_to_time(energy_peaks, sr=sr, hop_length=512)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
            
            silence_breaks: List[float] = []
            silence_threshold = np.percentile(energy, 20)
            for i in range(1, len(energy)):
                if energy[i-1] < silence_threshold and energy[i] > np.percentile(energy, 60):
                    silence_breaks.append(float(librosa.frames_to_time(i, sr=sr, hop_length=512)))
            
            time_steps = len(motion_scores)
            if len(energy) > 0 and time_steps > 0:
                audio_energy = np.interp(
                    np.linspace(0, len(energy) - 1, time_steps),
                    np.arange(len(energy)),
                    energy
                )
                audio_spectral = np.interp(
                    np.linspace(0, len(spectral_centroid) - 1, time_steps),
                    np.arange(len(spectral_centroid)),
                    spectral_centroid
                )
            else:
                audio_energy = np.zeros(time_steps)
                audio_spectral = np.zeros(time_steps)
            
            print(f"    ✓ Found {len(energy_peak_times)} audio peaks")
            
        except Exception as e:
            print(f"    ⚠️  Audio analysis limited: {e}")
            audio_energy = np.zeros(len(motion_scores))
            audio_spectral = np.zeros(len(motion_scores))
            energy_peak_times = np.array([])
            silence_breaks = []
        
        print("  🗣️  Transcribing with Whisper...")
        transcript_segments: List[Dict[str, Any]] = []
        contextual_moments: List[ContextualMoment] = []
        
        try:
            if USE_FASTER_WHISPER and self.whisper_model:
                segments, _ = self.whisper_model.transcribe(
                    video_path,
                    beam_size=5,
                    word_timestamps=True,
                    language='en'
                )
                
                for segment in segments:
                    text = segment.text.strip()
                    speech_analysis = self.analyze_speech_deep(text)
                    
                    duration_seg = segment.end - segment.start
                    word_count = len(text.split())
                    speech_rate = word_count / duration_seg if duration_seg > 0 else 0
                    
                    segment_data: Dict[str, Any] = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': text,
                        'words': [],
                        **speech_analysis,
                        'speech_rate': speech_rate
                    }
                    
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            segment_data['words'].append({
                                'text': word.word.strip(),
                                'start': word.start,
                                'end': word.end
                            })
                    
                    transcript_segments.append(segment_data)
                    
                    viral_score = (
                        len(speech_analysis['viral_triggers']) * 25 +
                        (100 if speech_analysis['sentiment'] == 'positive' else 50)
                    )
                    
                    if viral_score > 50 or len(speech_analysis['viral_triggers']) > 0:
                        visual_ctx = next(
                            (v for v in visual_contexts if abs(v['timestamp'] - segment.start) < 3),
                            {'scene': '', 'actions': [], 'objects': []}
                        )
                        
                        moment = ContextualMoment(
                            timestamp=segment.start,
                            duration=duration_seg,
                            speech_text=text,
                            speech_sentiment=speech_analysis['sentiment'],
                            speech_emotion=speech_analysis['emotion'],
                            entities=speech_analysis['entities'],
                            visual_scene=visual_ctx['scene'],
                            visual_objects=visual_ctx['objects'],
                            visual_actions=visual_ctx['actions'],
                            viral_score=viral_score,
                            viral_triggers=speech_analysis['viral_triggers'],
                            content_type=speech_analysis['content_type']
                        )
                        contextual_moments.append(moment)
            
            print(f"    ✓ Transcribed {len(transcript_segments)} segments")
            print(f"    ✓ Identified {len(contextual_moments)} viral moments")
            
        except Exception as e:
            print(f"    ⚠️  Transcription limited: {e}")
        
        all_text = " ".join([seg['text'] for seg in transcript_segments])
        overall_content = self.analyze_speech_deep(all_text) if all_text else {}
        
        total_analysis_time = time.time() - analysis_start
        print(f"\n  ⏱️  Total Analysis Time: {self.timer.format_time(total_analysis_time)}")
        print("=" * 80 + "\n")
        
        features = {
            'duration': duration,
            'fps': fps,
            'motion_scores': motion_scores,
            'audio_energy': audio_energy,
            'audio_spectral': audio_spectral,
            'face_presence': face_presence,
            'scene_changes': scene_changes,
            'energy_peaks': energy_peak_times,
            'silence_breaks': silence_breaks,
            'music_present': False,
            'transcript_segments': transcript_segments,
            'contextual_moments': contextual_moments,
            'visual_contexts': visual_contexts,
            'overall_content_type': overall_content.get('content_type', 'unknown'),
        }
        
        if self.analysis_cache:
            self.analysis_cache.save_analysis(video_path, features, total_analysis_time)
        
        return features
    
    def calculate_contextual_engagement(self, features: Dict[str, Any]) -> np.ndarray:
        print("📊 Calculating engagement scores...")
        
        motion = np.array(features['motion_scores'])
        audio = np.array(features['audio_energy'])
        spectral = np.array(features['audio_spectral'])
        faces = np.array(features['face_presence'])
        
        min_len = min(len(motion), len(audio), len(spectral), len(faces))
        motion = motion[:min_len]
        audio = audio[:min_len]
        spectral = spectral[:min_len]
        faces = faces[:min_len]
        
        scaler = StandardScaler()
        motion_norm = scaler.fit_transform(motion.reshape(-1, 1)).flatten()
        audio_norm = scaler.fit_transform(audio.reshape(-1, 1)).flatten()
        spectral_norm = scaler.fit_transform(spectral.reshape(-1, 1)).flatten()
        
        base_engagement = (
            0.15 * motion_norm +
            0.25 * audio_norm +
            0.15 * spectral_norm +
            0.15 * faces +
            0.30 * (motion_norm * audio_norm)
        )
        
        contextual_boost = np.zeros_like(base_engagement)
        
        for moment in features['contextual_moments']:
            idx = int(moment.timestamp)
            if idx < len(contextual_boost):
                boost_amount = moment.viral_score / 100
                contextual_boost[idx:min(idx + 5, len(contextual_boost))] += boost_amount
        
        engagement = 0.4 * base_engagement + 0.6 * contextual_boost
        
        window = 7
        engagement = np.convolve(engagement, np.ones(window)/window, mode='same')
        engagement = (engagement - engagement.min()) / (engagement.max() - engagement.min() + 1e-6) * 100
        
        print(f"  ✓ Calculated engagement scores\n")
        
        return engagement
    
    def find_optimal_clips(self, engagement: np.ndarray, features: Dict[str, Any],
                          n_clips: int = 3) -> List[Tuple[float, float, float, Dict[str, Any]]]:
        print("🎯 Finding optimal clips...")
        print("=" * 80)
        
        clip_len = self.clip_duration
        min_clip_len = max(30, clip_len - 15)
        max_clip_len = clip_len + 15
        
        candidates: List[Tuple[float, float, float, Dict[str, Any]]] = []
        video_duration = features['duration']
        
        for start_sec in range(0, int(video_duration - min_clip_len), 3):
            start_float = float(start_sec)
            end_float = min(start_float + clip_len, video_duration)
            
            actual_duration = end_float - start_float
            
            if actual_duration < min_clip_len or actual_duration > max_clip_len:
                continue
            
            start_idx = int(start_float)
            end_idx = int(end_float)
            
            if end_idx > len(engagement):
                continue
            
            avg_engagement = float(np.mean(engagement[start_idx:end_idx]))
            
            clip_moments = [
                m for m in features['contextual_moments']
                if start_float <= m.timestamp < end_float
            ]
            
            all_triggers: List[str] = []
            all_content_types: List[str] = []
            
            for moment in clip_moments:
                all_triggers.extend(moment.viral_triggers)
                if moment.content_type:
                    all_content_types.append(moment.content_type)
            
            content_type = max(set(all_content_types), key=all_content_types.count) if all_content_types else "unknown"
            viral_moment_count = len(clip_moments)
            unique_triggers = len(set(all_triggers))
            
            speech_coverage = 0.0
            for seg in features['transcript_segments']:
                overlap_start = max(start_float, seg['start'])
                overlap_end = min(end_float, seg['end'])
                if overlap_start < overlap_end:
                    speech_coverage += (overlap_end - overlap_start)
            
            speech_ratio = speech_coverage / actual_duration if actual_duration > 0 else 0
            
            viral_potential = (
                unique_triggers * 12 +
                viral_moment_count * 15
            )
            
            score = (
                0.30 * avg_engagement +
                0.30 * viral_potential +
                0.20 * speech_ratio * 100 +
                0.20 * 50
            )
            
            if speech_ratio < 0.3:
                score *= 0.5
            
            metadata: Dict[str, Any] = {
                'content_type': content_type,
                'viral_moments': viral_moment_count,
                'viral_triggers': list(set(all_triggers)),
                'speech_ratio': speech_ratio,
                'avg_engagement': avg_engagement,
                'viral_potential': viral_potential,
                'actual_duration': actual_duration,
            }
            
            candidates.append((start_float, end_float, score, metadata))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        selected: List[Tuple[float, float, float, Dict[str, Any]]] = []
        
        for clip in candidates:
            start, end, score, metadata = clip
            
            overlap = any(
                not (end <= s[0] + 5 or start >= s[1] - 5)
                for s in selected
            )
            
            if not overlap:
                selected.append((start, end, score, metadata))
                
                print(f"✓ Clip {len(selected)}: {start:.1f}-{end:.1f}s | Score: {score:.1f}")
                print(f"  • Type: {metadata['content_type']}")
                print(f"  • Triggers: {', '.join(metadata['viral_triggers'][:3]) if metadata['viral_triggers'] else 'None'}")
                print()
            
            if len(selected) >= n_clips:
                break
        
        print("=" * 80 + "\n")
        return sorted(selected, key=lambda x: x[0])


# ============================================================================
# ENHANCED SHORT CREATOR (Main Video Production)
# ============================================================================

class EnhancedShortCreator:
    """Complete short creation with all enhanced features"""
    
    def __init__(self, video_style: str = 'blurred', channel_name: str = "",
                 music_path: Optional[str] = None, music_volume: float = 0.3,
                 apply_shuffle: bool = True, apply_zoom: bool = True,
                 apply_audio_transforms: bool = True):
        
        self.video_style = video_style
        self.watermark_mgr = WatermarkManager(channel_name)
        self.music_mixer = BackgroundMusicMixer(music_path, music_volume)
        self.shuffler = SegmentShuffler() if apply_shuffle else None
        self.zoom_effect = SmoothZoomEffect() if apply_zoom else None
        self.audio_processor = AdvancedAudioProcessor() if apply_audio_transforms else None
        
        # Load Whisper for captions
        if USE_FASTER_WHISPER:
            if DEVICE == "cuda":
                self.whisper_model = WhisperModel("base", device="cuda", 
                                                 compute_type="float16", num_workers=4)
            else:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)
    
    def generate_captions(self, video_path: str, clip_start: float, clip_end: float) -> List[Dict[str, Any]]:
        """Generate captions for the clip"""
        print("💬 Generating captions...")
        
        try:
            words: List[Dict[str, Any]] = []
            
            if USE_FASTER_WHISPER and self.whisper_model:
                segments, _ = self.whisper_model.transcribe(
                    video_path,
                    beam_size=5,
                    word_timestamps=True,
                    language='en'
                )
                
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            # Check if word falls within clip bounds
                            if clip_start <= word.start < clip_end:
                                # Adjust timestamps relative to clip start
                                adjusted_start = max(0, word.start - clip_start)
                                adjusted_end = min(word.end - clip_start, clip_end - clip_start)
                                
                                words.append({
                                    'text': word.word.strip(),
                                    'start': adjusted_start,
                                    'end': adjusted_end
                                })
            else:
                result = self.whisper_model.transcribe(video_path, word_timestamps=True)
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_data in segment['words']:
                            if clip_start <= word_data['start'] < clip_end:
                                adjusted_start = max(0, word_data['start'] - clip_start)
                                adjusted_end = min(word_data['end'] - clip_start, clip_end - clip_start)
                                
                                words.append({
                                    'text': word_data['word'].strip(),
                                    'start': adjusted_start,
                                    'end': adjusted_end
                                })
            
            captions = self._group_words(words, max_words=2, max_duration=1.5)
            print(f"  ✓ Generated {len(captions)} captions")
            return captions
        except Exception as e:
            print(f"  ⚠️ Caption generation failed: {e}")
            return []
    
    def _group_words(self, words: List[Dict[str, Any]], max_words: int = 2, 
                     max_duration: float = 1.5) -> List[Dict[str, Any]]:
        if not words:
            return []
        
        grouped: List[Dict[str, Any]] = []
        current_group: List[str] = []
        current_start: Optional[float] = None
        current_end: float = 0.0
        
        for word in words:
            if not current_group:
                current_group = [word['text']]
                current_start = word['start']
                current_end = word['end']
            else:
                duration = word['end'] - (current_start or 0)
                if len(current_group) < max_words and duration < max_duration:
                    current_group.append(word['text'])
                    current_end = word['end']
                else:
                    grouped.append({
                        'text': ' '.join(current_group),
                        'start': current_start,
                        'end': current_end
                    })
                    current_group = [word['text']]
                    current_start = word['start']
                    current_end = word['end']
        
        if current_group:
            grouped.append({
                'text': ' '.join(current_group),
                'start': current_start,
                'end': current_end
            })
        
        return grouped
    
    def create_short(self, video_path: str, video_clip: VideoClip, captions: List[Dict],
                     output_path: str, features: Dict[str, Any] = None,
                     evasion: ContentIDEvasion = None, clip_start: float = 0, 
                     clip_end: float = None) -> str:
        """Create short with all enhancements"""
        
        print(f"\n{'='*80}")
        print(f"🎬 CREATING SHORT WITH ENHANCED FEATURES")
        print(f"{'='*80}")
        print(f"  Style: {VideoStyleMode.STYLES[self.video_style]['name']}")
        print(f"  Watermark: {self.watermark_mgr.channel_name or 'None'}")
        print(f"  Background Music: {'Yes' if self.music_mixer.music_path else 'No'}")
        print(f"  Segment Shuffle: {'Yes' if self.shuffler else 'No'}")
        print(f"  Smooth Zoom: {'Yes' if self.zoom_effect else 'No'}")
        print(f"  Audio Transforms: {'Yes' if self.audio_processor else 'No'}")
        print(f"{'='*80}\n")
        
        render_start = time.time()
        clip_duration = video_clip.duration
        
        # Step 1: Apply smooth zoom
        if self.zoom_effect:
            video_clip = self.zoom_effect.apply_zoom(video_clip)
        
        # Step 2: Extract audio for processing
        temp_audio = "temp_original_audio.wav"
        try:
            video_clip.audio.write_audiofile(temp_audio, logger=None, fps=44100)
        except:
            temp_audio = None
        
        # Step 3: Apply Content ID evasion if enabled
        if evasion and features and temp_audio:
            print("\n🛡️  Applying Content ID evasion...")
            mod_segments = evasion.identify_cuttable_segments(features, clip_duration)
            video_clip, temp_audio = evasion.apply_smart_modifications(
                video_clip, mod_segments, temp_audio
            )
        
        # Step 4: Apply segment shuffle (if enabled)
        if self.shuffler and temp_audio:
            video_clip, temp_audio = self.shuffler.shuffle_segments(
                video_clip, temp_audio, preserve_speech=True
            )
        
        # Step 5: Apply audio transforms
        if self.audio_processor and temp_audio:
            temp_audio = self.audio_processor.apply_advanced_transforms(temp_audio)
        
        # Step 6: Mix background music
        if temp_audio:
            speech_times = [(c['start'], c['end']) for c in captions if 'start' in c and 'end' in c]
            temp_audio = self.music_mixer.mix_audio(
                temp_audio, clip_duration, speech_timestamps=speech_times
            )
            
            # Set mixed audio
            mixed_audio_clip = AudioFileClip(temp_audio)
            if mixed_audio_clip.duration > video_clip.duration:
                mixed_audio_clip = mixed_audio_clip.subclip(0, video_clip.duration)
            video_clip = video_clip.set_audio(mixed_audio_clip)
        
        # Step 7: Create vertical format with chosen style
        orig_w, orig_h = video_clip.size
        target_w, target_h = 1080, 1920
        
        # Resize video to fit
        scale = min(target_w / orig_w, target_h / orig_h)
        video_resized = video_clip.resize(scale)
        new_w, new_h = video_resized.size
        
        # Create background based on style
        background_clip = VideoStyleMode.create_background(
            self.video_style, video_clip, (target_w, target_h)
        )
        
        # Position video in center
        x_pos = (target_w - new_w) // 2
        y_pos = (target_h - new_h) // 2
        
        if self.video_style == 'split':
            y_pos = int(target_h * 0.15)
        
        video_resized = video_resized.set_position((x_pos, y_pos))
        
        # Step 8: Add captions
        text_clips = self._create_caption_clips(captions, target_w, target_h)
        
        # Step 9: Add watermarks
        watermark_clips = self.watermark_mgr.create_watermark_clips(
            video_clip.duration, (target_w, target_h)
        )
        
        # Step 10: Composite everything
        all_clips = [background_clip, video_resized] + text_clips + watermark_clips
        final_video = CompositeVideoClip(all_clips, size=(target_w, target_h))
        final_video = final_video.set_audio(video_clip.audio)
        
        # Step 11: Render with enhanced encoding
        print("🎞️  Rendering with enhanced encoding...")
        try:
            bitrate = random.choice(['4500k', '5000k', '5500k'])
            keyframe_interval = random.randint(24, 48)
            
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                preset='medium',
                bitrate=bitrate,
                audio_bitrate='192k',
                threads=4,
                logger=None,
                ffmpeg_params=[
                    '-g', str(keyframe_interval),
                    '-sc_threshold', '0',
                    '-bf', '2',
                ]
            )
            
            render_time = time.time() - render_start
            file_size = os.path.getsize(output_path) / (1024*1024)
            
            print(f"\n✅ Short created successfully!")
            print(f"  • Output: {output_path}")
            print(f"  • Size: {file_size:.1f} MB")
            print(f"  • Render time: {TimeEstimator.format_time(render_time)}")
            print(f"  • Encoding: {bitrate} bitrate, keyframe every {keyframe_interval} frames")
            
        finally:
            final_video.close()
            background_clip.close()
            video_resized.close()
            for clip in text_clips + watermark_clips:
                try:
                    clip.close()
                except:
                    pass
            
            # Cleanup temp files
            for temp_file in [temp_audio, "temp_shuffled_audio.wav", 
                            "temp_transformed_audio.wav", "temp_mixed_audio.wav"]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            clear_gpu_memory()
        
        return output_path
    
    def _create_caption_clips(self, captions: List[Dict], 
                             target_w: int, target_h: int) -> List[ImageClip]:
        """Create caption clips based on style - positioned in lower third"""
        
        if not captions:
            return []
        
        text_clips = []
        
        # Font sizes optimized for lower third positioning
        if self.video_style == 'meme':
            font_size = 55
            position_y = 50  # Top for meme style
        elif self.video_style == 'split':
            font_size = 48
            position_y = int(target_h * 0.75)
        else:
            font_size = 65
            # Position in lower third (around 62-65% down the screen)
            # This matches the brown box area in your reference image
            position_y = int(target_h * 0.62)  # Lower third area
        
        font = self._load_font(font_size)
        
        for caption in captions:
            if 'text' not in caption:
                continue
            
            text = caption['text'].upper()
            
            img = Image.new('RGBA', (target_w, 300), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (target_w - text_width) // 2
            y = 50
            
            # Reduced outline width
            outline_w = 6  # Reduced from 10
            for dx in range(-outline_w, outline_w + 1, 2):
                for dy in range(-outline_w, outline_w + 1, 2):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill='#000000')
            
            draw.text((x, y), text, font=font, fill='#FFFFFF')
            
            duration = caption.get('end', 1) - caption.get('start', 0)
            
            # Ensure duration is positive and reasonable
            if duration <= 0:
                duration = 0.5
            
            caption_clip = ImageClip(np.array(img), duration=duration, transparent=True)
            caption_clip = caption_clip.set_start(caption.get('start', 0))
            caption_clip = caption_clip.set_position(('center', position_y))
            text_clips.append(caption_clip)
        
        return text_clips
    
    def _load_font(self, size: int) -> Any:
        font_paths = [
            "C:\\Windows\\Fonts\\impact.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except:
                    continue
        
        return ImageFont.load_default()


# ============================================================================
# YOUTUBE UPLOADER
# ============================================================================

class YouTubeUploader:
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    
    def __init__(self, credentials_file: str = 'client_secrets.json') -> None:
        self.credentials_file = credentials_file
        self.youtube: Any = self._get_authenticated_service()
    
    def _get_authenticated_service(self) -> Any:
        creds: Optional[Credentials] = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        return build('youtube', 'v3', credentials=creds)
    
    def upload_video(self, video_path: str, title: str, description: str,
                    tags: List[str], category_id: str = '22') -> str:
        print(f"📤 Uploading: {title}")
        upload_start = time.time()
        
        if '#Shorts' not in title and '#shorts' not in title:
            title = f"{title} #Shorts"
        
        body = {
            'snippet': {
                'title': title[:100],
                'description': description[:5000],
                'tags': tags + ['shorts', 'youtubeshorts', 'viral'],
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': 'public',
                'selfDeclaredMadeForKids': False
            }
        }
        
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = self.youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        response = request.execute()
        video_id = response['id']
        
        upload_time = time.time() - upload_start
        print(f"✅ Uploaded! https://youtube.com/shorts/{video_id} ({TimeEstimator.format_time(upload_time)})\n")
        return video_id


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(video_input: str, n_clips: int = 3, clip_duration: int = 60,
         upload_to_youtube: bool = False, use_learning: bool = True, 
         apply_anti_copyright: bool = True, video_style: str = 'blurred',
         channel_name: str = "", music_path: Optional[str] = None,
         music_volume: float = 0.3, apply_shuffle: bool = True,
         apply_zoom: bool = True, apply_audio_transforms: bool = True) -> None:
    
    main_timer = TimeEstimator()
    main_timer.start()
    
    print("\n" + "=" * 80)
    print("🧠 ULTIMATE AI SHORTS CREATOR - ENHANCED EDITION")
    print("=" * 80 + "\n")
    
    # Initialize analyzer
    analyzer = AdvancedAIAnalyzer(
        clip_duration=clip_duration, 
        use_learning=use_learning, 
        apply_anti_copyright=apply_anti_copyright
    )
    
    # Initialize enhanced creator
    creator = EnhancedShortCreator(
        video_style=video_style,
        channel_name=channel_name,
        music_path=music_path,
        music_volume=music_volume,
        apply_shuffle=apply_shuffle,
        apply_zoom=apply_zoom,
        apply_audio_transforms=apply_audio_transforms
    )
    
    uploader: Optional[YouTubeUploader] = YouTubeUploader() if upload_to_youtube else None
    
    # Get video
    video_path, video_title = analyzer.get_video_path(video_input)
    is_local_file = os.path.exists(video_input) or os.path.exists(os.path.join(os.getcwd(), "videos", video_input))
    cleanup_video = not is_local_file
    
    # Analyze video
    features = analyzer.analyze_video_contextual(video_path)
    
    # Calculate engagement
    engagement = analyzer.calculate_contextual_engagement(features)
    
    # Find best clips
    best_clips = analyzer.find_optimal_clips(engagement, features, n_clips=n_clips)
    
    if not best_clips:
        print("❌ No suitable clips found.")
        return
    
    print(f"🎬 Creating {len(best_clips)} Shorts...\n")
    print("=" * 80 + "\n")
    
    created_shorts: List[Dict[str, Any]] = []
    
    for i, (start, end, score, metadata) in enumerate(best_clips):
        clip_timer_start = time.time()
        
        print(f"{'='*80}")
        print(f"PROCESSING CLIP {i+1}/{len(best_clips)}")
        print(f"{'='*80}")
        print(f"  Time: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
        print(f"  Score: {score:.1f}/100")
        print(f"  Type: {metadata['content_type']}")
        print(f"{'='*80}\n")
        
        # Extract clip
        clip_path = f"clip_{i+1}_temp.mp4"
        video = VideoFileClip(video_path).subclip(start, end)
        video.write_videofile(clip_path, codec='libx264', audio_codec='aac',
                            logger=None, preset='ultrafast')
        video.close()
        
        # Generate captions
        captions = creator.generate_captions(video_path, start, end)
        
        # Create final short
        output_path = f"short_{i+1}_final.mp4"
        video_clip = VideoFileClip(clip_path)
        
        creator.create_short(
            video_path, video_clip, captions, output_path,
            features=features if apply_anti_copyright else None,
            evasion=analyzer.evasion if apply_anti_copyright else None,
            clip_start=start,
            clip_end=end
        )
        video_clip.close()
        
        # Log performance
        record_id = None
        if analyzer.performance_tracker and use_learning:
            record_id = analyzer.performance_tracker.log_short_creation(metadata, output_path)
        
        created_shorts.append({
            'path': output_path,
            'metadata': metadata,
            'score': score,
            'start': start,
            'end': end,
            'record_id': record_id
        })
        
        # Upload if requested
        if uploader:
            title = f"{video_title[:40]} - {metadata['content_type'][:20]} #{i+1}"
            description = (
                f"🔥 AI-Selected Viral Moment\n\n"
                f"📊 Score: {score:.1f}/100\n"
                f"• Type: {metadata['content_type']}\n"
                f"\n#Shorts #Viral #Trending"
            )
            
            tags = ['viral', 'trending', 'ai', 'shorts'] + metadata['viral_triggers'][:3]
            video_id = uploader.upload_video(output_path, title, description, tags)
            
            if record_id:
                print(f"\n  💡 Track performance: Update record ID '{record_id}' with YouTube analytics")
        
        # Cleanup
        try:
            os.remove(clip_path)
        except Exception:
            pass
        
        clear_gpu_memory()
    
    # Cleanup downloaded video
    if cleanup_video:
        try:
            os.remove(video_path)
        except Exception:
            pass
    
    total_time = main_timer.get_elapsed()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 ALL SHORTS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n⏱️  TOTAL TIME: {total_time}")
    print(f"Total Shorts: {len(created_shorts)}")
    
    if apply_anti_copyright:
        print(f"\n🛡️  CONTENT ID PROTECTION APPLIED")
    
    if use_learning:
        print(f"\n📊 PERFORMANCE TRACKING ENABLED")
    
    print("\nRanked by Viral Potential:\n")
    
    ranked = sorted(created_shorts, key=lambda x: x['score'], reverse=True)
    for idx, short in enumerate(ranked, 1):
        size_mb = os.path.getsize(short['path']) / (1024*1024)
        duration = short['metadata'].get('actual_duration', '?')
        print(f"{idx}. {short['path']} ({size_mb:.1f} MB, {duration:.1f}s)")
        print(f"   Score: {short['score']:.1f} | Type: {short['metadata']['content_type']}")
        if short['record_id']:
            print(f"   🆔 Record ID: {short['record_id']}")
        print()
    
    print("=" * 80 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧠 ULTIMATE AI SHORTS CREATOR - ENHANCED EDITION v2.0")
    print("=" * 80)
    print("\n🚀 FEATURES:")
    print("  ✓ 5 Video Style Modes (Blurred, Black, White, Meme, Split)")
    print("  ✓ Channel Watermarks (Top-right & Bottom-left)")
    print("  ✓ Background Music with Auto-Ducking")
    print("  ✓ Segment Jitter & Shuffle (Copyright Evasion)")
    print("  ✓ Smooth Random Zoom Effects")
    print("  ✓ Advanced Audio Transforms (Pitch/Speed/EQ)")
    print("  ✓ Content ID Evasion (Smart Jump Cuts)")
    print("  ✓ GPU-Accelerated AI Analysis")
    print("  ✓ Dataset Learning & Performance Tracking")
    print("  ✓ Enhanced Re-encoding (Random Bitrates/Keyframes)")
    print("=" * 80)
    
    VIDEO_INPUT = input("\n🎥 Enter YouTube URL or video path: ").strip()
    
    # Video style selection
    print("\n📺 SELECT VIDEO STYLE:")
    for i, (key, info) in enumerate(VideoStyleMode.STYLES.items(), 1):
        print(f"  {i}. {info['name']} - {info['description']}")
    
    style_choice = input("Choose style (1-5, default 1): ").strip() or "1"
    style_map = {str(i+1): key for i, key in enumerate(VideoStyleMode.STYLES.keys())}
    video_style = style_map.get(style_choice, 'blurred')
    
    # Channel watermark
    channel_name = input("\n🏷️  Enter channel name for watermark (or press Enter to skip): ").strip()
    
    # Background music
    music_choice = input("\n🎵 Add background music? (yes/no, default no): ").lower()
    music_path = None
    music_volume = 0.3
    
    if music_choice == 'yes':
        music_path = input("   Enter music file path: ").strip()
        if music_path and os.path.exists(music_path):
            volume_input = input("   Music volume (0.0-1.0, default 0.3): ").strip()
            try:
                music_volume = float(volume_input) if volume_input else 0.3
                music_volume = max(0.0, min(1.0, music_volume))
            except:
                music_volume = 0.3
        else:
            print("   ⚠️ Music file not found, skipping")
            music_path = None
    
    # Basic settings
    NUM_CLIPS = int(input("\n📊 Number of Shorts (1-5, default 3): ") or "3")
    CLIP_DURATION = int(input("⏱️  Duration per Short (30-90s, default 60): ") or "60")
    
    # Enhancement toggles
    apply_shuffle = input("\n🔀 Apply segment shuffle? (yes/no, default yes): ").lower() != 'no'
    apply_zoom = input("🔍 Apply smooth zoom? (yes/no, default yes): ").lower() != 'no'
    apply_audio_transforms = input("🎚️ Apply audio transforms? (yes/no, default yes): ").lower() != 'no'
    
    # Advanced settings
    learning_choice = input("\n🧠 Enable dataset learning? (yes/no, default yes): ").lower()
    USE_LEARNING = learning_choice != 'no'
    
    copyright_choice = input("🛡️ Enable Content ID evasion? (yes/no, default yes): ").lower()
    APPLY_ANTI_COPYRIGHT = copyright_choice != 'no'
    
    upload_choice = input("📤 Upload to YouTube? (yes/no, default no): ").lower()
    UPLOAD = upload_choice == 'yes'
    
    print(f"\n{'='*80}")
    print("⚙️  CONFIGURATION SUMMARY:")
    print(f"  • Style: {VideoStyleMode.STYLES[video_style]['name']}")
    print(f"  • Watermark: {'@' + channel_name if channel_name else 'None'}")
    print(f"  • Music: {os.path.basename(music_path) if music_path else 'None'}")
    if music_path:
        print(f"    - Volume: {music_volume * 100:.0f}%")
    print(f"  • Clips: {NUM_CLIPS} x {CLIP_DURATION}s")
    print(f"  • Shuffle: {apply_shuffle}")
    print(f"  • Zoom: {apply_zoom}")
    print(f"  • Audio Transforms: {apply_audio_transforms}")
    print(f"  • Learning: {USE_LEARNING}")
    print(f"  • Content ID Evasion: {APPLY_ANTI_COPYRIGHT}")
    print(f"  • Upload: {UPLOAD}")
    print('='*80 + "\n")
    
    try:
        main(
            VIDEO_INPUT, 
            n_clips=NUM_CLIPS, 
            clip_duration=CLIP_DURATION,
            upload_to_youtube=UPLOAD, 
            use_learning=USE_LEARNING,
            apply_anti_copyright=APPLY_ANTI_COPYRIGHT,
            video_style=video_style,
            channel_name=channel_name,
            music_path=music_path,
            music_volume=music_volume,
            apply_shuffle=apply_shuffle,
            apply_zoom=apply_zoom,
            apply_audio_transforms=apply_audio_transforms
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()