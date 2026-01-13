import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FluteSound:
    """Generate realistic flute sounds from notes."""

    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.A4 = 440.0

    def note_to_freq(self, note_name: str, octave: int) -> float:
        """Convert note name and octave to frequency."""
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

        # Handle Indian notation (simplified for this context, can be expanded)
        sargam_map = {'Sa': 0, 'r': 1, 'R': 2, 'g': 3, 'G': 4, 'm': 5,
                      'M': 6, 'P': 7, 'd': 8, 'D': 9, 'n': 10, 'N': 11}

        if note_name in note_map:
            semitones = note_map[note_name]
        elif note_name in sargam_map:
            semitones = sargam_map[note_name]
        else:
            semitones = 0 # Default to C

        # Calculate frequency
        midi_note = 12 * (octave + 1) + semitones
        # Adjust MIDI note for A4=69, so C4=60
        frequency = self.A4 * (2 ** ((midi_note - 69) / 12))
        return frequency

    def generate_flute_tone(self, frequency: float, duration: float,
                           attack=0.05, decay=0.1, sustain=0.7, release=0.15) -> np.ndarray:
        """
        Generate a realistic flute tone using additive synthesis.

        Args:
            frequency: Note frequency in Hz
            duration: Duration in seconds
            attack, decay, sustain, release: ADSR envelope parameters
        """
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)

        # Flute has strong fundamental and odd harmonics (like organ pipes)
        # Harmonic structure: 1, 2 (weak), 3, 4 (very weak), 5, etc.
        harmonics = [
            (1.0, 1.0),    # Fundamental (strong)
            (2.0, 0.3),    # 2nd harmonic (weak for flute)
            (3.0, 0.4),    # 3rd harmonic
            (4.0, 0.15),   # 4th harmonic (very weak)
            (5.0, 0.2),    # 5th harmonic
            (6.0, 0.08),   # 6th harmonic
            (7.0, 0.1),    # 7th harmonic
        ]

        # Generate harmonics
        audio = np.zeros_like(t)
        for harmonic_num, amplitude in harmonics:
            # Add slight detuning for realism
            detune = np.random.uniform(-0.5, 0.5)
            freq = frequency * harmonic_num + detune
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.5

        # Apply ADSR envelope
        envelope = self._adsr_envelope(len(audio), attack, decay, sustain, release, duration)
        audio = audio * envelope

        # Add breath noise (characteristic of flute)
        breath_noise = np.random.normal(0, 0.02, len(audio))
        breath_noise = signal.lfilter([1], [1, -0.95], breath_noise)  # Low-pass filter
        audio += breath_noise * envelope * 0.3

        # Apply a gentle vibrato
        vibrato_rate = 5.5  # Hz
        vibrato_depth = 0.015  # 1.5% frequency modulation
        # This vibrato is applied to the frequency, not by resampling, for simplicity
        # A more accurate vibrato would modulate the phase over time.
        # For now, let's simplify to avoid complex resampling logic here.
        # Instead of recalculating all harmonics with vibrato, just blend the original with a vibrato-effected version
        # This is a simplification but gets some vibrato effect across.
        t_vibrato = np.linspace(0, duration, len(audio), endpoint=False)
        vibrato_signal = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t_vibrato)
        vibrato_audio = np.zeros_like(t_vibrato)
        for harmonic_num, amplitude in harmonics:
            vibrato_audio += amplitude * np.sin(2 * np.pi * frequency * harmonic_num * vibrato_signal * t_vibrato)

        vibrato_audio = vibrato_audio / (np.max(np.abs(vibrato_audio)) + 1e-8) * 0.5
        audio = audio * 0.7 + vibrato_audio * envelope * 0.3 # Blend original with vibrato part

        return audio

    def _adsr_envelope(self, length: int, attack: float, decay: float,
                       sustain: float, release: float, total_duration: float) -> np.ndarray:
        """Generate ADSR envelope."""
        envelope = np.zeros(length)

        # Calculate sample points
        attack_samples = int(attack * self.sr)
        decay_samples = int(decay * self.sr)
        release_samples = int(release * self.sr)
        
        # Ensure non-negative and correct sum
        if attack_samples + decay_samples + release_samples > length:
            # Scale all down if too long
            scale = length / (attack_samples + decay_samples + release_samples)
            attack_samples = int(attack_samples * scale)
            decay_samples = int(decay_samples * scale)
            release_samples = int(release_samples * scale)

        sustain_samples = length - attack_samples - decay_samples - release_samples

        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples, endpoint=False)

        # Decay
        start_idx = attack_samples
        end_idx = start_idx + decay_samples
        envelope[start_idx:end_idx] = np.linspace(1, sustain, decay_samples, endpoint=False)

        # Sustain
        start_idx = end_idx
        end_idx = start_idx + sustain_samples
        envelope[start_idx:end_idx] = sustain

        # Release
        start_idx = end_idx
        envelope[start_idx:] = np.linspace(sustain, 0, release_samples)

        return envelope

    def generate_from_notes(self, notes: List[Dict], output_file: str = None) -> np.ndarray:
        """
        Generate audio from a list of note dictionaries.

        Args:
            notes: List of note dicts with 'note', 'start', 'duration', etc.
            output_file: Optional path to save audio

        Returns:
            Generated audio as numpy array
        """
        if not notes:
            return np.array([])

        # Calculate total duration
        # Ensure we don't end up with negative duration or too short a note
        valid_notes = [n for n in notes if n.get('duration', 0) > 0 and n.get('start', 0) >= 0]
        if not valid_notes:
            return np.array([])

        total_duration = max(n.get('end', n['start'] + n['duration']) for n in valid_notes)
        total_samples = int(total_duration * self.sr) + self.sr # Add 1 second padding for release tails
        audio = np.zeros(total_samples)

        print(f"\n🎵 Synthesizing {len(valid_notes)} notes...")

        for i, note_info in enumerate(valid_notes):
            # Extract note information
            note_name = ''
            octave = 4 # Default octave

            # Prioritize 'svara_name' and 'octave_shift' if present
            if 'svara_name' in note_info and 'octave_shift' in note_info:
                note_name = note_info['svara_name']
                # Octave shift is relative to Sa's octave. For synthesis, we need absolute octave.
                # Assuming Sa (C4) is the reference, so octave_shift 0 means C4-B4 range, +1 means C5-B5 etc.
                # For simplicity in this synthesis, let's map it roughly.
                # A more precise mapping would convert Svara to Western note, then get absolute octave.
                # For now, let's use a simple mapping:
                octave = 4 + note_info['octave_shift']

            elif 'note_data' in note_info:
                note_name = note_info['note_data'].get('western', 'C')
                octave = note_info['note_data'].get('octave', 4)
            elif 'note' in note_info: # Fallback for old structure
                note_str = note_info['note']
                if note_str[-1].isdigit():
                    note_name = note_str[:-1]
                    octave = int(note_str[-1])
                else:
                    note_name = 'C'
                    octave = 4

            # If note_name is still empty, default
            if not note_name:
                note_name = 'Sa'

            frequency = self.note_to_freq(note_name, octave)
            duration = note_info.get('duration', 0.5)
            start_time = note_info.get('start', 0)

            # Ensure duration is positive
            if duration <= 0:
                continue

            # Generate tone
            tone = self.generate_flute_tone(frequency, duration)

            # Add to audio at correct position
            start_sample = int(start_time * self.sr)
            end_sample = start_sample + len(tone)

            # Pad audio array if needed (should be handled by initial total_samples calc)
            if end_sample > len(audio):
                audio = np.pad(audio, (0, end_sample - len(audio)))

            audio[start_sample:end_sample] += tone

            if (i + 1) % 10 == 0:
                print(f"  Synthesized {i + 1}/{len(valid_notes)} notes...")

        # Normalize final audio again
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 1e-8: # Avoid division by zero
                audio = audio / max_val * 0.8
            else:
                audio = np.zeros_like(audio)

        # Save if output file specified
        if output_file:
            try:
                sf.write(output_file, audio, self.sr)
                print(f"✅ Synthesized audio saved to: {output_file}")
            except Exception as e:
                print(f"Error saving synthesized audio to {output_file}: {e}")

        return audio


class AudioComparator:
    """Compare two audio signals for similarity."""

    def __init__(self):
        pass

    def compute_spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray, sr: int) -> float:
        """
        Compute spectral similarity between two audio signals.
        Returns score between 0 and 1 (1 = identical).
        """
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        if min_len == 0: return 0.0

        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        # Compute spectrograms
        # Using default n_fft=2048, hop_length=512 for STFT
        try:
            S1 = np.abs(librosa.stft(audio1))
            S2 = np.abs(librosa.stft(audio2))
        except Exception as e:
            print(f"Error during STFT: {e}")
            return 0.0

        # Normalize
        S1 = S1 / (np.max(S1) + 1e-8)
        S2 = S2 / (np.max(S2) + 1e-8)

        # Flatten for correlation
        flat_S1 = S1.flatten()
        flat_S2 = S2.flatten()

        if len(flat_S1) < 2 or len(flat_S2) < 2: # np.corrcoef needs at least 2 points
            return 0.0

        # Compute correlation
        correlation = np.corrcoef(flat_S1, flat_S2)[0, 1]

        return max(0, correlation)  # Ensure non-negative

    def compute_pitch_similarity(self, notes1: List[Dict], notes2: List[Dict]) -> float:
        """
        Compare pitch sequences between original detection and synthesis.
        This needs to be more robust than just correlation of raw frequencies.
        """
        if not notes1 or not notes2:
            return 0.0

        # Extract note labels (svara_full or note string) for comparison
        labels1 = [n.get('svara_full', n.get('note', '')) for n in notes1]
        labels2 = [n.get('svara_full', n.get('note', '')) for n in notes2]

        # Calculate agreement rate
        match_count = 0
        min_len = min(len(labels1), len(labels2))

        for i in range(min_len):
            if labels1[i] == labels2[i]:
                match_count += 1

        return match_count / min_len if min_len > 0 else 0.0

    def compute_rhythm_similarity(self, notes1: List[Dict], notes2: List[Dict]) -> float:
        """Compare rhythmic patterns based on event sequences and durations."""
        if not notes1 or not notes2: return 0.0

        # Compare sequence of events (start times and durations)
        total_duration = max(n.get('end', 0) for n in notes1 + notes2) # Max duration to normalize by
        if total_duration == 0: return 0.0

        # Create event arrays for onset times
        onsets1 = np.array([n.get('start', 0) for n in notes1])
        onsets2 = np.array([n.get('start', 0) for n in notes2])

        # Compare number of notes
        num_notes_sim = 1 - (abs(len(notes1) - len(notes2)) / max(len(notes1), len(notes2), 1))

        # Compare cumulative durations (simplified rhythmic contour)
        cum_dur1 = np.cumsum([n.get('duration', 0) for n in notes1])
        cum_dur2 = np.cumsum([n.get('duration', 0) for n in notes2])

        min_cum_len = min(len(cum_dur1), len(cum_dur2))
        if min_cum_len > 0:
            rhythm_contour_sim = np.corrcoef(cum_dur1[:min_cum_len], cum_dur2[:min_cum_len])[0,1]
            rhythm_contour_sim = max(0, rhythm_contour_sim) # ensure non-negative
        else:
            rhythm_contour_sim = 0.0

        # Combine metrics
        # Weights can be tuned. For now, simple average
        similarity = (num_notes_sim + rhythm_contour_sim) / 2.0

        return max(0, min(1, similarity)) # Clamp between 0 and 1

    def compute_overall_similarity(self, audio1: np.ndarray, audio2: np.ndarray,
                                  notes1: List[Dict], notes2: List[Dict], sr: int) -> Dict:
        """
        Compute comprehensive similarity metrics.
        """
        spectral = self.compute_spectral_similarity(audio1, audio2, sr)
        pitch = self.compute_pitch_similarity(notes1, notes2)
        rhythm = self.compute_rhythm_similarity(notes1, notes2)

        # Weighted average (weights can be tuned based on what's most important)
        overall = 0.5 * spectral + 0.3 * pitch + 0.2 * rhythm

        return {
            'overall': overall,
            'spectral': spectral,
            'pitch': pitch,
            'rhythm': rhythm
        }


class FeedbackTranscriptionEngine:
    """Main engine with feedback loop for improved accuracy."""

    def __init__(self, base_note='Sa', max_iterations=5, target_similarity=0.85):
        self.base_note = base_note
        self.max_iterations = max_iterations
        self.target_similarity = target_similarity
        self.sr = 22050

        self.synthesizer = FluteSound(sample_rate=self.sr)
        self.comparator = AudioComparator()

        # For pitch detection (using the svara map from previous steps for consistency)
        self.A4 = 440.0
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.svara_map = [
            'Sa',  # C
            'r',   # C# / Komal Re
            'R',   # D  / Shuddh Re
            'g',   # D# / Komal Ga
            'G',   # E  / Shuddh Ga
            'm',   # F  / Shuddh Ma
            'M',   # F# / Teevra Ma
            'P',   # G  / Pa
            'd',   # G# / Komal Dha
            'D',   # A  / Shuddh Dha
            'n',   # A# / Komal Ni
            'N'    # B  / Shuddh Ni
        ]
        self.sa_freq_tonic = 261.63 # Default Sa to C4

    def freq_to_svara(self, frequency: float, sa_freq: float) -> Tuple[str, int, float]:
        """
        Convert frequency to Indian Svara name, octave, and cents deviation.
        (Copied from FluteNotationEngine to be self-contained)
        """
        if frequency < 20: # Too low for meaningful Svara
            return None, None, None

        # Calculate MIDI note number for the input frequency
        midi_note_input = 69 + 12 * np.log2(frequency / self.A4)

        # Calculate MIDI note number for Sa
        sa_midi_note = 69 + 12 * np.log2(sa_freq / self.A4)

        # Determine the relative semitone index from Sa (0 to 11)
        relative_midi_note = midi_note_input - sa_midi_note

        # Find the nearest semitone (rounded MIDI note) relative to Sa
        midi_note_relative_to_sa = round(relative_midi_note)

        svara_index = int(midi_note_relative_to_sa % 12)
        if svara_index < 0:
            svara_index += 12

        svara_name = self.svara_map[svara_index]

        # Calculate the ideal frequency of this svara based on sa_freq
        ideal_svara_freq = sa_freq * (2**(midi_note_relative_to_sa / 12))

        # Calculate cents deviation from the ideal svara frequency
        cents_off = 1200 * np.log2(frequency / ideal_svara_freq)

        # Determine the octave relative to the Sa's octave. (0 for Sa's octave, 1 for higher, -1 for lower)
        octave_shift = int(np.floor(relative_midi_note / 12))

        return svara_name, octave_shift, cents_off

    def is_valid_flute_note(self, frequency: float) -> bool:
        # Flute range: C4 (261.63 Hz) to C7 (2093.00 Hz)
        # Extended for safety: B3 to D7
        min_freq = 246.94  # B3
        max_freq = 2349.32  # D7
        return min_freq <= frequency <= max_freq

    def initial_transcription(self, audio_path: str) -> Tuple[List[Dict], np.ndarray]:
        """Perform initial transcription of audio, outputting Svaras."""
        print(f"\n📊 Initial transcription of: {audio_path}")

        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        # Pitch detection using pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('B3'),
            fmax=librosa.note_to_hz('D7'),
            sr=sr,
            frame_length=2048,
            hop_length=512
        )

        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

        # Process detected pitches into Svaras
        svaras = []
        current_svara_full = None
        note_start = None
        last_cents = 0.0 # To store cents deviation for the last processed frame

        for time, freq, is_voiced, conf in zip(times, f0, voiced_flag, voiced_probs):
            if is_voiced and not np.isnan(freq) and conf > 0.5 and self.is_valid_flute_note(freq):
                svara_name, octave_shift, cents = self.freq_to_svara(freq, sa_freq=self.sa_freq_tonic)
                if svara_name is None: continue # Skip if freq_to_svara failed

                svara_full = f"{svara_name}{octave_shift}"
                last_cents = cents

                if current_svara_full != svara_full:
                    if current_svara_full is not None:
                        svaras.append({
                            'svara_full': current_svara_full,
                            'svara_name': current_svara_full[:-1],
                            'octave_shift': int(current_svara_full[-1]),
                            'start': note_start,
                            'end': time,
                            'duration': time - note_start,
                            'cents_deviation': last_cents # Using last cents of the segment
                        })
                    current_svara_full = svara_full
                    note_start = time
            else:
                if current_svara_full is not None:
                    svaras.append({
                        'svara_full': current_svara_full,
                        'svara_name': current_svara_full[:-1],
                        'octave_shift': int(current_svara_full[-1]),
                        'start': note_start,
                        'end': time,
                        'duration': time - note_start,
                        'cents_deviation': last_cents
                    })
                    current_svara_full = None
                    note_start = None

        # Add the last note if the audio ended on a note
        if current_svara_full is not None:
            svaras.append({
                'svara_full': current_svara_full,
                'svara_name': current_svara_full[:-1],
                'octave_shift': int(current_svara_full[-1]),
                'start': note_start,
                'end': times[-1] if times.size > 0 else note_start + 0.1, # ensure end is > start
                'duration': (times[-1] if times.size > 0 else note_start + 0.1) - note_start,
                'cents_deviation': last_cents
            })

        print(f"  Detected {len(svaras)} Svaras")
        return svaras, y

    def refine_notes(self, notes: List[Dict], similarity_scores: Dict) -> List[Dict]:
        """
        Refine note detection based on similarity feedback.
        This is a heuristic-based refinement. In a real system, this would involve
        re-evaluating pitch detection parameters or applying ML-based corrections.
        """
        refined_notes = []

        # Simple refinement strategies based on overall similarity
        if similarity_scores['overall'] < 0.6: # Significant mismatch
            # Adjust durations more aggressively for better rhythmic fit
            for note in notes:
                new_duration = note['duration']
                if similarity_scores['rhythm'] < 0.7: # If rhythm is an issue
                    if new_duration < 0.1: new_duration *= 1.2 # Make very short notes slightly longer
                    elif new_duration > 1.0: new_duration *= 0.9 # Make very long notes slightly shorter
                refined_notes.append({
                    **note, 
                    'duration': new_duration
                })

        elif similarity_scores['overall'] < 0.8: # Moderate mismatch
            # Apply minor adjustments
            for note in notes:
                new_duration = note['duration']
                if similarity_scores['rhythm'] < 0.8:
                    new_duration *= np.random.uniform(0.95, 1.05) # Small random duration jitter
                refined_notes.append({
                    **note, 
                    'duration': new_duration
                })
        else: # High similarity, make minimal changes or none
            refined_notes = notes.copy()

        return refined_notes

    def transcribe_with_feedback(self, audio_path: str, output_prefix: str = "output") -> Dict:
        """
        Main transcription with feedback loop.

        Returns:
            Dict with final notes, synthesized audio, and metrics
        """
        print("\n" + "="*70)
        print("FEEDBACK-BASED TRANSCRIPTION SYSTEM (Indian Svaras)")
        print("="*70)

        # Load original audio
        original_audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        # Initial transcription
        notes, _ = self.initial_transcription(audio_path)

        best_notes = notes
        best_similarity = 0.0
        iteration_history = []

        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            print("-" * 70)

            if not notes: # If no notes detected, cannot synthesize or compare
                print("No notes to process in this iteration. Breaking loop.")
                break

            # Synthesize audio from current notes
            synth_file = f"{output_prefix}_synth_iter{iteration + 1}.wav"
            synthesized_audio = self.synthesizer.generate_from_notes(notes, synth_file)

            # Ensure synthesized_audio is not empty before comparing
            if len(synthesized_audio) == 0:
                print("Synthesized audio is empty. Cannot compare. Breaking loop.")
                break

            # Compare with original
            similarity_scores = self.comparator.compute_overall_similarity(
                original_audio, synthesized_audio, notes, notes, self.sr
            )

            print(f"\n📈 Similarity Scores:")
            print(f"  Overall:   {similarity_scores['overall']:.2%}")
            print(f"  Spectral:  {similarity_scores['spectral']:.2%}")
            print(f"  Pitch:     {similarity_scores['pitch']:.2%}")
            print(f"  Rhythm:    {similarity_scores['rhythm']:.2%}")

            iteration_history.append({
                'iteration': iteration + 1,
                'similarity': similarity_scores,
                'num_notes': len(notes)
            })

            # Update best if improved
            if similarity_scores['overall'] > best_similarity:
                best_similarity = similarity_scores['overall']
                best_notes = notes.copy()
                print(f"  ✅ New best similarity: {best_similarity:.2%}")

            # Check if target reached
            if similarity_scores['overall'] >= self.target_similarity:
                print(f"\n🎯 Target similarity reached!")
                break

            # Refine notes for next iteration
            if iteration < self.max_iterations - 1:
                notes = self.refine_notes(notes, similarity_scores)
            else:
                print("Max iterations reached. No further refinement.")

        # Generate final synthesized audio from the best notes found
        print(f"\n🎵 Generating final synthesized audio from best notes...")
        final_synth_file = f"{output_prefix}_final_synthesis.wav"
        final_audio = self.synthesizer.generate_from_notes(best_notes, final_synth_file)

        # Save results
        results = {
            'notes': best_notes,
            'final_similarity': best_similarity,
            'iterations_run': len(iteration_history),
            'iteration_history': iteration_history,
            'synthesized_audio_file': final_synth_file,
            'total_notes_in_best_transcription': len(best_notes),
            'total_duration': best_notes[-1]['end'] if best_notes else 0
        }

        results_file = f"{output_prefix}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✅ TRANSCRIPTION COMPLETE")
        print(f"{'='*70}")
        print(f"Final similarity: {best_similarity:.2%}")
        print(f"Total iterations run: {len(iteration_history)}")
        print(f"Total notes in best transcription: {len(best_notes)}")
        print(f"Synthesized audio: {final_synth_file}")
        print(f"Results JSON: {results_file}")
        print(f"{'='*70}\n")

        return results


def test_feedback_transcription(audio_file: str, output_prefix: str = "output"):
    """
    Test the feedback-based transcription system.
    """
    engine = FeedbackTranscriptionEngine(
        base_note='Sa', # Using 'Sa' as the base for Indian Svara mapping
        max_iterations=5,
        target_similarity=0.85
    )

    results = engine.transcribe_with_feedback(audio_file, output_prefix)

    return results

# Call the function for this specific subtask
results = test_feedback_transcription('/content/O Palan Hare - Flute _ Instrumental.mp3', 'feedback_notes')
print("Feedback transcription completed. Check the generated files for results.")
