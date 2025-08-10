#!/usr/bin/env python3

# midi_visualizer.py
#
# This version fixes the piano key highlighting to match the active note's color.
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

# --- Configuration ---
CONFIG = {
    "video": {"width": 1920, "height": 1080, "fps": 30},
    "visualization": {
        "fall_time_seconds": 4.0,
        "note_colors": [
            (52, 152, 219), (231, 76, 60), (46, 204, 113),
            (241, 196, 15), (155, 89, 182), (26, 188, 156), (230, 126, 34),
        ],
        "background_color": (30, 30, 30),
    },
    "piano": {
        "white_key_color": (248, 249, 249), "black_key_color": (23, 32, 42),
        "key_highlight_color": (52, 152, 219), # Fallback, no longer primary
        "height_percentage": 0.15,
    }
}

# --- Helper Functions ---
def is_black_key(midi_note: int) -> bool:
    return (midi_note % 12) in {1, 3, 6, 8, 10}

def check_command_availability(command: str) -> bool:
    return shutil.which(command) is not None

# --- Data Structures ---
@dataclass
class Note:
    pitch: int; velocity: int; start_time: float; end_time: float; track_index: int
    @property
    def duration(self) -> float: return self.end_time - self.start_time

# --- Core Logic ---
class MidiVisualizer:
    def __init__(self, midi_files: List[str], output_path: str, audio_path: str = None):
        self.midi_files = midi_files
        self.output_path = output_path
        self.audio_path = audio_path
        self.notes: List[Note] = []
        self.video_duration = 0.0
        self._check_system_dependencies()

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.Surface((CONFIG['video']['width'], CONFIG['video']['height']))
        self.piano_layout = self._get_piano_layout()

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

    def _draw_frame(self, current_time: float):
        """Draws a single frame, now with color-correct key highlighting."""
        w, h = self.screen.get_size()
        self.screen.fill(CONFIG['visualization']['background_color'])
        piano_h = h * CONFIG['piano']['height_percentage']
        render_h = h - piano_h

        # --- KEY HIGHLIGHTING FIX ---
        # Instead of a set of active pitches, we create a dictionary mapping
        # a pitch to its active color. This way we know what color to use.
        active_pitch_colors = {}
        colors = CONFIG['visualization']['note_colors']
        num_colors = len(colors)
        
        # Since self.notes is sorted by start_time, iterating through it and
        # overwriting the color ensures the key takes the color of the newest note.
        for n in self.notes:
            if n.start_time <= current_time < n.end_time:
                active_pitch_colors[n.pitch] = colors[n.track_index % num_colors]
            # Small optimization: no need to check notes that haven't started yet
            if n.start_time > current_time:
                break
        
        # Draw piano keys using the new color mapping
        for note_pitch, rect in self.piano_layout['white_keys']:
            # Use .get() to provide a default color if the key is not active
            color = active_pitch_colors.get(note_pitch, CONFIG['piano']['white_key_color'])
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
        
        for note_pitch, rect in self.piano_layout['black_keys']:
            color = active_pitch_colors.get(note_pitch, CONFIG['piano']['black_key_color'])
            pygame.draw.rect(self.screen, color, rect)

        # Draw falling notes (no changes here)
        fall_time = CONFIG['visualization']['fall_time_seconds']
        pixels_per_sec = render_h / fall_time
        num_white_keys = len(self.piano_layout['white_keys'])
        white_key_width = w / num_white_keys
        white_note_width = white_key_width * 0.9
        black_note_width = white_key_width * 0.55

        for note in self.notes:
            if note.end_time < current_time or note.start_time > current_time + fall_time: continue
            y_bottom = render_h - (note.start_time - current_time) * pixels_per_sec
            y_top = render_h - (note.end_time - current_time) * pixels_per_sec
            y_draw_top, y_draw_bottom = max(0, y_top), min(render_h, y_bottom)

            if y_draw_bottom > y_draw_top:
                note_width = black_note_width if is_black_key(note.pitch) else white_note_width
                center_x = self.piano_layout['key_x_positions'].get(note.pitch, 0)
                note_rect = pygame.Rect(center_x - note_width / 2, y_draw_top, note_width, y_draw_bottom - y_draw_top)
                color = colors[note.track_index % num_colors]
                pygame.draw.rect(self.screen, color, note_rect)

    def _make_frame_for_moviepy(self, t: float):
        self._draw_frame(t)
        frame_data = pygame.surfarray.pixels3d(self.screen)
        return frame_data.swapaxes(0, 1)

    def render(self):
        start_time = time.time()
        self._load_and_process_midi()
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

#21