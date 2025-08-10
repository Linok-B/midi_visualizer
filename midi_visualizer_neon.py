#!/usr/bin/env python3

# midi_visualizer.py
#
# Final version with refined glow effects and a darker keyboard aesthetic.
#

import pretty_midi
import numpy as np
import pygame
from moviepy.editor import VideoClip, AudioFileClip, CompositeAudioClip
import argparse
import os
import tempfile
import time
import sys
import shutil
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import random

# --- Configuration ---
CONFIG = {
    "video": {"width": 1920, "height": 1080, "fps": 30},
    "visualization": {
        "fall_time_seconds": 4.0,
        "note_colors": [
            (0, 255, 255), (255, 0, 255), (50, 255, 50),
            (255, 255, 0), (255, 100, 255), (100, 100, 255),
            (255, 128, 0),
        ],
        "background_color": (10, 10, 15),
    },
    "piano": {
        # Final adjustment for a darker, more subtle keyboard
        "white_key_color": (80, 80, 85),
        "black_key_color": (30, 30, 35),
        "height_percentage": 0.15,
    },
    "particles": {
        "enabled": True, "count_per_hit": 25, "gravity": 400, "max_lifespan_ms": 1200,
        "min_velocity_x": -250, "max_velocity_x": 250,
        "min_velocity_y": -400, "max_velocity_y": -150,
    },
    "glow": {
        "enabled": True,
        "intensity": 4,
        "downscale_factor": 6,
    }
}

# --- Helper Functions & Data Structures ---
def is_black_key(midi_note: int) -> bool:
    return (midi_note % 12) in {1, 3, 6, 8, 10}

def check_command_availability(command: str) -> bool:
    return shutil.which(command) is not None

@dataclass(unsafe_hash=True)
class Note:
    pitch: int; velocity: int; start_time: float; end_time: float; track_index: int
    @property
    def duration(self) -> float: return self.end_time - self.start_time

@dataclass
class Particle:
    x: float; y: float; vx: float; vy: float; color: tuple
    lifespan: float; max_lifespan: float; size: float

# --- Core Logic ---
class MidiVisualizer:
    def __init__(self, midi_files: List[str], output_path: str, audio_path: str = None):
        self.midi_files = midi_files
        self.output_path = output_path
        self.audio_path = audio_path
        self.notes: List[Note] = []
        self.notes_by_pitch: Dict[int, List[Note]] = defaultdict(list)
        self.video_duration = 0.0
        self._check_system_dependencies()

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.Surface((CONFIG['video']['width'], CONFIG['video']['height']))
        self.piano_layout = self._get_piano_layout()
        
        self.particles: List[Particle] = []
        self.triggered_notes: set = set()

        if CONFIG['glow']['enabled']:
            self.glow_source_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

    def _check_system_dependencies(self):
        missing_deps = [cmd for cmd in ["ffmpeg", "fluidsynth"] if not check_command_availability(cmd)]
        if missing_deps:
            raise RuntimeError(f"Missing system dependencies: {', '.join(missing_deps)}. Please install them.")

    def _load_and_process_midi(self):
        print("Processing MIDI files with pretty_midi library...")
        all_notes = []
        total_duration = 0.0

        for i, file_path in enumerate(self.midi_files):
            try:
                print(f"-> Loading file {i+1}: {file_path}")
                midi_data = pretty_midi.PrettyMIDI(file_path)
                total_duration = max(total_duration, midi_data.get_end_time())
                for instrument in midi_data.instruments:
                    if instrument.is_drum: instrument.program = 0
                    for note in instrument.notes:
                        all_notes.append(Note(
                            pitch=note.pitch, velocity=note.velocity,
                            start_time=note.start, end_time=note.end, track_index=i
                        ))
            except Exception as e:
                print(f"Warning: Could not process MIDI file {file_path}. Error: {e}", file=sys.stderr)

        if not all_notes: raise ValueError("No valid notes found in MIDI files.")
        self.notes = sorted(all_notes, key=lambda n: n.start_time)

        for note in self.notes:
            self.notes_by_pitch[note.pitch].append(note)

        self.video_duration = total_duration + 1.5
        print(f"Found {len(self.notes)} notes from {len(self.midi_files)} file(s). Video duration set to: {self.video_duration:.2f} seconds.")


    def _prepare_audio(self) -> str:
        if self.audio_path:
            if not os.path.exists(self.audio_path): raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            print(f"Using provided audio file: {self.audio_path}")
            return self.audio_path

        soundfont_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
        if not os.path.exists(soundfont_path):
             raise FileNotFoundError(f"Soundfont not found at {soundfont_path}. Please ensure 'fluid-soundfont-gm' is installed.")

        if len(self.midi_files) == 1:
            print(f"Synthesizing audio from '{self.midi_files[0]}' using FluidSynth...")
            temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_audio_fd)
            try:
                from midi2audio import FluidSynth
                FluidSynth(sound_font=soundfont_path).midi_to_audio(self.midi_files[0], temp_audio_path)
                return temp_audio_path
            except Exception as e:
                if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                raise
        
        print(f"Multiple MIDI files detected. Synthesizing and mixing audio for {len(self.midi_files)} files...")
        intermediate_wavs = []
        try:
            from midi2audio import FluidSynth
            fs = FluidSynth(sound_font=soundfont_path)
            for i, midi_file in enumerate(self.midi_files):
                temp_fd, temp_path = tempfile.mkstemp(suffix=f"_{i}.wav")
                os.close(temp_fd)
                print(f"  -> Synthesizing {os.path.basename(midi_file)}...")
                fs.midi_to_audio(midi_file, temp_path)
                intermediate_wavs.append(temp_path)

            print("  -> Mixing audio tracks with MoviePy...")
            audio_clips = [AudioFileClip(f) for f in intermediate_wavs]
            mixed_audio = CompositeAudioClip(audio_clips)
            final_fd, final_path = tempfile.mkstemp(suffix="_mixed.wav")
            os.close(final_fd)
            mixed_audio.write_audiofile(final_path, codec='pcm_s16le', fps=44100, logger=None)
            for clip in audio_clips: clip.close()
            return final_path

        finally:
            for f in intermediate_wavs:
                if os.path.exists(f): os.remove(f)

    def _get_piano_layout(self) -> Dict:
        layout = {'white_keys': [], 'black_keys': [], 'key_x_positions': {}}
        w, h = CONFIG['video']['width'], CONFIG['video']['height']
        piano_h = h * CONFIG['piano']['height_percentage']
        white_key_notes = [n for n in range(21, 109) if not is_black_key(n)]
        white_w = w / len(white_key_notes)
        black_w, black_h = white_w * 0.6, piano_h * 0.6
        white_idx = 0
        for note in range(21, 109):
            if not is_black_key(note):
                x = white_idx * white_w
                layout['white_keys'].append((note, pygame.Rect(x, h - piano_h, white_w, piano_h)))
                layout['key_x_positions'][note] = x + (white_w / 2)
                white_idx += 1
        for note in range(21, 109):
            if is_black_key(note):
                x_center = layout['key_x_positions'].get(note - 1, 0) - (white_w / 2) + white_w
                layout['black_keys'].append((note, pygame.Rect(x_center - (black_w / 2), h - piano_h, black_w, black_h)))
                layout['key_x_positions'][note] = x_center
        return layout

    def _create_particle_burst(self, note: Note, x_pos: float, y_pos: float):
        p_conf = CONFIG['particles']
        color = CONFIG['visualization']['note_colors'][note.track_index % len(CONFIG['visualization']['note_colors'])]
        for _ in range(p_conf['count_per_hit']):
            lifespan = random.uniform(p_conf['max_lifespan_ms'] * 0.5, p_conf['max_lifespan_ms']) / 1000.0
            self.particles.append(Particle(
                x=x_pos, y=y_pos,
                vx=random.uniform(p_conf['min_velocity_x'], p_conf['max_velocity_x']),
                vy=random.uniform(p_conf['min_velocity_y'], p_conf['max_velocity_y']),
                color=color, lifespan=lifespan, max_lifespan=lifespan, size=random.uniform(1, 4)
            ))

    def _update_and_draw_particles(self, dt: float):
        p_conf = CONFIG['particles']
        gravity = p_conf['gravity']
        
        for i in range(len(self.particles) - 1, -1, -1):
            particle = self.particles[i]
            particle.lifespan -= dt
            if particle.lifespan <= 0:
                self.particles.pop(i)
                continue
            
            particle.vy += gravity * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt

            size = particle.size * (particle.lifespan / particle.max_lifespan)
            if size > 1:
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, particle.color, (size, size), size)
                self.screen.blit(particle_surf, (particle.x - size, particle.y - size), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_glowing_note(self, surface: pygame.Surface, rect: pygame.Rect, color: tuple):
        half_width = rect.width / 2
        if half_width <= 0: return

        for i in range(int(rect.width)):
            x = rect.left + i
            brightness = 1.0 - abs(i - half_width) / half_width
            brightness = brightness ** 2

            line_color = (
                int(color[0] * brightness),
                int(color[1] * brightness),
                int(color[2] * brightness)
            )
            pygame.draw.line(surface, line_color, (x, rect.top), (x, rect.bottom))

    def _draw_frame(self, current_time: float):
        w, h = self.screen.get_size()
        piano_h = h * CONFIG['piano']['height_percentage']
        render_h = h - piano_h
        colors = CONFIG['visualization']['note_colors']
        glow_conf = CONFIG['glow']

        active_notes_by_pitch = defaultdict(list)
        for n in self.notes:
            if n.start_time > current_time: break
            if n.start_time <= current_time < n.end_time:
                active_notes_by_pitch[n.pitch].append(n)

        self.screen.fill(CONFIG['visualization']['background_color'])
        for note_pitch, rect in self.piano_layout['white_keys']:
            pygame.draw.rect(self.screen, CONFIG['piano']['white_key_color'], rect)
            pygame.draw.rect(self.screen, (20, 20, 25), rect, 1) # Very dark outlines
        for note_pitch, rect in self.piano_layout['black_keys']:
            pygame.draw.rect(self.screen, CONFIG['piano']['black_key_color'], rect)

        if glow_conf['enabled']:
            self.glow_source_surface.fill((0, 0, 0, 0))

            def draw_split_key_glow(key_rect, notes_on_key, surface):
                num_notes = len(notes_on_key)
                if num_notes == 0: return
                split_width = key_rect.width / num_notes
                for i, note in enumerate(sorted(notes_on_key, key=lambda n: n.track_index)):
                    color = colors[note.track_index % len(colors)]
                    split_rect = pygame.Rect(key_rect.left + i * split_width, key_rect.top, split_width, key_rect.height)
                    pygame.draw.rect(surface, color, split_rect)

            for note_pitch, rect in self.piano_layout['white_keys']:
                if active_notes_by_pitch[note_pitch]:
                    draw_split_key_glow(rect, active_notes_by_pitch[note_pitch], self.glow_source_surface)
            for note_pitch, rect in self.piano_layout['black_keys']:
                 if active_notes_by_pitch[note_pitch]:
                    draw_split_key_glow(rect, active_notes_by_pitch[note_pitch], self.glow_source_surface)

            fall_time = CONFIG['visualization']['fall_time_seconds']
            pixels_per_sec = render_h / fall_time
            white_key_width = w / len(self.piano_layout['white_keys'])
            
            visible_notes = [n for n in self.notes if not (n.end_time < current_time or n.start_time > current_time + fall_time)]
            for note in visible_notes:
                overlapping_notes = [o for o in self.notes_by_pitch[note.pitch] if note.start_time < o.end_time and note.end_time > o.start_time]
                overlapping_notes.sort(key=lambda n: n.track_index)
                try:
                    note_index = overlapping_notes.index(note)
                    num_splits = len(overlapping_notes)
                except ValueError: continue
                
                base_width = white_key_width * (0.55 if is_black_key(note.pitch) else 0.9)
                split_width = base_width / num_splits
                center_x = self.piano_layout['key_x_positions'].get(note.pitch, 0)
                base_left_x = center_x - (base_width / 2)
                note_x = base_left_x + (note_index * split_width)
                
                y_bottom = render_h - (note.start_time - current_time) * pixels_per_sec
                y_top = render_h - (note.end_time - current_time) * pixels_per_sec
                
                if min(render_h, y_bottom) > max(0, y_top):
                    note_rect = pygame.Rect(note_x, max(0, y_top), split_width, min(render_h, y_bottom) - max(0, y_top))
                    note_color = colors[note.track_index % len(colors)]
                    self._draw_glowing_note(self.glow_source_surface, note_rect, note_color)

            low_res_size = (w // glow_conf['downscale_factor'], h // glow_conf['downscale_factor'])
            glow_surface_blurred = pygame.transform.smoothscale(self.glow_source_surface, low_res_size)
            for _ in range(glow_conf['intensity']):
                glow_surface_blurred = pygame.transform.smoothscale(glow_surface_blurred, (w,h))
            self.screen.blit(glow_surface_blurred, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            
            self.screen.blit(self.glow_source_surface, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

        if CONFIG['particles']['enabled']:
            for n in self.notes:
                note_key = (n.start_time, n.pitch, n.track_index)
                if n.start_time <= current_time < n.end_time and note_key not in self.triggered_notes:
                    center_x = self.piano_layout['key_x_positions'].get(n.pitch, 0)
                    self._create_particle_burst(n, center_x, render_h)
                    self.triggered_notes.add(note_key)
            dt = 1.0 / CONFIG['video']['fps']
            self._update_and_draw_particles(dt)

    def _make_frame_for_moviepy(self, t: float):
        self._draw_frame(t)
        frame_copy = self.screen.copy()
        frame_data = pygame.surfarray.pixels3d(frame_copy)
        return frame_data.swapaxes(0, 1)

    def render(self):
        start_time = time.time()
        self._load_and_process_midi()
        self.triggered_notes.clear()
        
        temp_audio_path = None
        try:
            temp_audio_path = self._prepare_audio()
            print("\nAssembling video with MoviePy (on-demand frame rendering)...")
            video_clip = VideoClip(make_frame=self._make_frame_for_moviepy, duration=self.video_duration)
            video_clip.fps = CONFIG['video']['fps']
            audio_clip = AudioFileClip(temp_audio_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(
                self.output_path, codec='libx264', audio_codec='aac',
                threads=os.cpu_count(), preset='medium', logger='bar'
            )
            pygame.quit()
        finally:
            if temp_audio_path and self.audio_path is None and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"\nCleaned up temporary audio file: {temp_audio_path}")
        print(f"\nSuccessfully created video: {self.output_path}")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Creates a piano-roll video from MIDI files.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('midi_files', nargs='+', help="Path to one or more MIDI files. Each file will be a different color.")
    parser.add_argument('-o', '--output-file', default='output.mp4', help="Output video file path (default: output.mp4).")
    parser.add_argument('-a', '--audio-file', help="Optional audio file for the soundtrack.\nIf not provided, audio is synthesized from the provided MIDI file(s).")
    args = parser.parse_args()
    for f in args.midi_files:
        if not os.path.exists(f):
            print(f"Error: Input MIDI file not found at '{f}'", file=sys.stderr)
            sys.exit(1)
    try:
        MidiVisualizer(args.midi_files, args.output_file, args.audio_file).render()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

#37