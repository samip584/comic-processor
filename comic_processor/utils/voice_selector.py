"""Interactive voice selector with preview functionality and preset inference."""

import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Union
import sys


class VoiceSelector:
    """Interactive voice selection with audio preview + preset inference.

    quick_select returns (voice_path, preset).
    Preset heuristic:
      storyteller/warm/energetic -> expressive
      dramatic/intense/epic      -> emotional
      neutral/clear              -> stable_clone
      else                       -> demo_default
    """

    def __init__(self, voice_samples_dir: Optional[Path] = None) -> None:
        if voice_samples_dir is None:
            project_root = Path(__file__).parent.parent.parent
            voice_samples_dir = project_root / "voice_samples"
        self.voice_samples_dir = voice_samples_dir
        self.available_voices = self._scan_voices()

    def _scan_voices(self) -> List[Tuple[Path, str]]:
        if not self.voice_samples_dir.exists():
            return []
        voices: List[Tuple[Path, str]] = []
        for voice_file in sorted(self.voice_samples_dir.glob("*.wav")):
            display_name = voice_file.stem.replace("voice_", "").replace("_", " ").title()
            voices.append((voice_file, display_name))
        return voices

    def _play_audio(self, audio_path: Path) -> bool:
        try:
            players = ["afplay", "aplay", "ffplay", "play"]
            for player in players:
                try:
                    check = subprocess.run(["which", player], capture_output=True, timeout=1)
                    if check.returncode == 0:
                        if player == "ffplay":
                            subprocess.run([player, "-nodisp", "-autoexit", str(audio_path)], capture_output=True, timeout=30)
                        else:
                            subprocess.run([player, str(audio_path)], capture_output=True, timeout=30)
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            print("   ⚠️  No audio player found (afplay/aplay/ffplay/play)")
            return False
        except Exception as e:
            print(f"   ⚠️  Could not play audio: {e}")
            return False

    def select_voice(self) -> Optional[str]:
        if not self.available_voices:
            print("\n⚠️  No voice samples found!")
            print(f"   Looking in: {self.voice_samples_dir}")
            print("\n💡 Run 'python generate_styletts2_voices.py' to generate voices")
            print("   Or the system will use StyleTTS 2 default voice.\n")
            return None

        print("\n" + "=" * 70)
        print("🎙️  NARRATOR VOICE SELECTION")
        print("=" * 70)
        print(f"\nFound {len(self.available_voices)} voice sample(s):\n")
        for idx, (voice_path, display_name) in enumerate(self.available_voices, 1):
            print(f"{idx}. {display_name}")
            print(f"   └─ {voice_path.name}")
        print(f"\n{len(self.available_voices) + 1}. [Auto] Use default voice")
        print("\n" + "=" * 70)
        print("\n💡 Commands:")
        print("   • Enter number to select voice style")
        print("   • Type 'p <number>' to preview")
        print("   • Press Enter for default (#1)")
        print("   • Type 'q' to quit")
        print("\n" + "=" * 70 + "\n")

        while True:
            try:
                choice = input("Your choice [1]: ").strip()
                
                # Default to first voice if empty
                if not choice:
                    selected_path, display_name = self.available_voices[0]
                    print(f"\n✅ Selected: {display_name}")
                    print(f"   File: {selected_path.name}\n")
                    return str(selected_path)
                
                choice_lower = choice.lower()
                
                if choice_lower == "q":
                    print("\n👋 Exiting...\n")
                    sys.exit(0)
                    
                if choice_lower.startswith("p "):
                    try:
                        num = int(choice.split()[1])
                        if 1 <= num <= len(self.available_voices):
                            voice_path, display_name = self.available_voices[num - 1]
                            print(f"\n🔊 Playing: {display_name}")
                            if not self._play_audio(voice_path):
                                print(f"   📂 {voice_path}")
                            print()
                        else:
                            print(f"❌ Invalid number. Choose 1-{len(self.available_voices)}")
                    except (ValueError, IndexError):
                        print("❌ Invalid preview command. Use: p <number>")
                    continue
                    
                try:
                    num = int(choice)
                    if 1 <= num <= len(self.available_voices):
                        selected_path, display_name = self.available_voices[num - 1]
                        print(f"\n✅ Selected: {display_name}")
                        print(f"   File: {selected_path.name}\n")
                        return str(selected_path)
                    elif num == len(self.available_voices) + 1:
                        print("\n✅ Using auto-download mode (default narrator)\n")
                        return None
                    else:
                        print(f"❌ Enter a number 1-{len(self.available_voices) + 1}")
                except ValueError:
                    print("❌ Invalid input. Enter number, 'p <n>', or press Enter for default.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\n⚠️  Input interrupted → using first voice\n")
                if self.available_voices:
                    selected_path, display_name = self.available_voices[0]
                    print(f"✅ Auto-selected: {display_name}\n")
                    return str(selected_path)
                return None

    @staticmethod
    def quick_select(include_preset: bool = True) -> Union[str, tuple]:
        selector = VoiceSelector()
        voice_path = selector.select_voice()
        if not include_preset:
            return voice_path
        preset = "demo_default"
        if voice_path:
            lname = Path(voice_path).name.lower()
            if any(k in lname for k in ["dramatic", "intense", "epic"]):
                preset = "emotional"
            elif any(k in lname for k in ["storyteller", "warm", "energetic"]):
                preset = "expressive"
            elif any(k in lname for k in ["neutral", "clear"]):
                preset = "stable_clone"
        print(f"🗣️ Voice: {voice_path or 'default narrator'} | Preset: {preset}")
        return voice_path, preset


if __name__ == "__main__":
    vp, preset = VoiceSelector.quick_select()
    print(f"Result: voice={vp}, preset={preset}")
