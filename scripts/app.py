"""Gradio web UI for Beat Saber map generation.

Upload a song, pick difficulty/genre, and generate a playable .zip.
Includes links to ArcViewer and BS Map Check for previewing.

Usage:
    python scripts/app.py
    python scripts/app.py --port 7860 --share
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = REPO_ROOT / "data" / "generated"
CHECKPOINTS_DIR = REPO_ROOT / "outputs"

DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
GENRES = [
    "unknown", "electronic", "rock", "pop", "anime",
    "hip-hop", "classical", "jazz", "country", "video-game", "other",
]

# Preview tool URLs
ARCVIEWER_URL = "https://allpoland.github.io/ArcViewer/"
MAPCHECK_URL = "https://kivalevan.me/BeatSaber-MapCheck/"
PARITY_URL = "https://galaxymaster2.github.io/bs-parity/"


def _find_checkpoints() -> dict[str, Path | None]:
    """Scan outputs/ for the best checkpoint per stage."""
    best: dict[str, Path | None] = {"onset": None, "sequence": None, "lighting": None}

    if not CHECKPOINTS_DIR.exists():
        return best

    for ckpt in CHECKPOINTS_DIR.rglob("*.ckpt"):
        name = ckpt.stem.lower()
        if name == "last":
            # Use last.ckpt as fallback — check sibling ckpts for stage
            siblings = list(ckpt.parent.glob("*.ckpt"))
            for sib in siblings:
                if sib.stem.startswith("onset"):
                    if best["onset"] is None:
                        best["onset"] = ckpt
                elif sib.stem.startswith("sequence"):
                    if best["sequence"] is None:
                        best["sequence"] = ckpt
                elif sib.stem.startswith("lighting"):
                    if best["lighting"] is None:
                        best["lighting"] = ckpt
        elif name.startswith("onset"):
            # Pick the one with highest val_f1
            best["onset"] = _pick_better(best["onset"], ckpt, "onset")
        elif name.startswith("sequence"):
            best["sequence"] = _pick_better(best["sequence"], ckpt, "sequence")
        elif name.startswith("lighting"):
            best["lighting"] = _pick_better(best["lighting"], ckpt, "lighting")

    return best


def _pick_better(
    current: Path | None, candidate: Path, stage: str
) -> Path:
    """Pick the better checkpoint based on filename metrics."""
    if current is None:
        return candidate
    # For onset, higher val_f1 is better; for others, lower val_loss is better
    # Just use the most recently modified file as heuristic
    if candidate.stat().st_mtime > current.stat().st_mtime:
        return candidate
    return current


def generate_map(
    audio_file: str | None,
    difficulties: list[str],
    genre: str,
    bpm: float | None,
    song_name: str,
    song_author: str,
    onset_threshold: float,
    temperature: float,
    beam_size: int,
    use_sampling: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str]:
    """Generate a Beat Saber map from uploaded audio.

    Returns:
        Tuple of (path to .zip file for download, status message).
    """
    if audio_file is None:
        return None, "Please upload an audio file first."

    if not difficulties:
        return None, "Please select at least one difficulty."

    audio_path = Path(audio_file)
    progress(0.1, desc="Loading models...")

    # Find best checkpoints
    ckpts = _find_checkpoints()
    ckpt_status = []
    for stage, path in ckpts.items():
        if path:
            ckpt_status.append(f"{stage}: {path.name}")
        else:
            ckpt_status.append(f"{stage}: untrained (random weights)")

    progress(0.2, desc="Loading audio and detecting BPM...")

    # Import here to avoid slow startup
    from beatsaber_automapper.generation.generate import generate_level

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output name
    name = song_name.strip() if song_name.strip() else audio_path.stem
    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in name)
    output_path = OUTPUTS_DIR / f"{safe_name}.zip"

    progress(0.3, desc="Running onset detection (Stage 1)...")

    try:
        result = generate_level(
            audio_path=audio_path,
            output_path=output_path,
            difficulties=difficulties,
            onset_checkpoint=ckpts["onset"],
            sequence_checkpoint=ckpts["sequence"],
            lighting_checkpoint=ckpts["lighting"],
            onset_threshold=onset_threshold,
            temperature=temperature,
            beam_size=beam_size,
            use_sampling=use_sampling,
            song_name=name,
            song_author=song_author.strip() or "Unknown Artist",
            bpm=bpm if bpm and bpm > 0 else None,
            genre=genre,
        )
        progress(1.0, desc="Done!")

        status_lines = [
            f"Generated: {result.name}",
            f"Difficulties: {', '.join(difficulties)}",
            f"Genre: {genre}",
            "",
            "Checkpoints used:",
            *[f"  {s}" for s in ckpt_status],
            "",
            "Preview your map:",
            f"  ArcViewer: {ARCVIEWER_URL}",
            f"  BS Map Check: {MAPCHECK_URL}",
            f"  Parity Check: {PARITY_URL}",
            "",
            "Drag the downloaded .zip into ArcViewer to preview!",
        ]
        return str(result), "\n".join(status_lines)

    except Exception as e:
        logger.exception("Generation failed")
        return None, f"Generation failed: {e}"


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    # Check for available checkpoints at startup
    ckpts = _find_checkpoints()
    ckpt_info = []
    for stage in ("onset", "sequence", "lighting"):
        if ckpts[stage]:
            ckpt_info.append(f"  {stage}: {ckpts[stage].name}")
        else:
            ckpt_info.append(f"  {stage}: not found (will use random weights)")

    with gr.Blocks(
        title="Beat Saber Automapper",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Beat Saber Automapper\n"
            "Generate playable Beat Saber levels from any song. "
            "Upload audio, configure settings, and download a .zip "
            "you can preview in "
            "[ArcViewer](https://allpoland.github.io/ArcViewer/)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Upload Song",
                    type="filepath",
                    sources=["upload"],
                )
                with gr.Row():
                    song_name = gr.Textbox(
                        label="Song Name",
                        placeholder="(auto-detected from filename)",
                    )
                    song_author = gr.Textbox(
                        label="Artist",
                        placeholder="Unknown Artist",
                    )
                with gr.Row():
                    diff_select = gr.CheckboxGroup(
                        choices=DIFFICULTIES,
                        value=["Expert"],
                        label="Difficulties",
                    )
                    genre_select = gr.Dropdown(
                        choices=GENRES,
                        value="unknown",
                        label="Genre",
                    )

            with gr.Column(scale=1):
                bpm_input = gr.Number(
                    label="BPM (0 = auto-detect)",
                    value=0,
                    precision=1,
                )
                onset_thresh = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Onset Threshold (lower = more notes)",
                )
                temp_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature (higher = more varied)",
                )
                beam_input = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=8,
                    step=1,
                    label="Beam Size",
                )
                sampling_check = gr.Checkbox(
                    label="Use Nucleus Sampling (instead of beam search)",
                    value=False,
                )

        generate_btn = gr.Button(
            "Generate Beat Saber Map",
            variant="primary",
            size="lg",
        )

        with gr.Row():
            output_file = gr.File(
                label="Download Generated Map (.zip)",
            )
            status_box = gr.Textbox(
                label="Status",
                lines=12,
                interactive=False,
            )

        with gr.Accordion("Model Checkpoints", open=False):
            gr.Markdown(
                "The app auto-detects the best checkpoints from `outputs/`.\n\n"
                "Current checkpoints:\n"
                + "\n".join(ckpt_info)
                + "\n\nTrain models with "
                "`python scripts/run_training_pipeline.py` to improve quality."
            )

        with gr.Accordion("Preview Tools", open=False):
            gr.Markdown(
                "After downloading, preview your map with these tools:\n\n"
                f"- **[ArcViewer]({ARCVIEWER_URL})** — "
                "3D preview with game-accurate visuals. "
                "Drag your .zip file in.\n"
                f"- **[BS Map Check]({MAPCHECK_URL})** — "
                "Validates structure, flags errors and warnings.\n"
                f"- **[Parity Checker]({PARITY_URL})** — "
                "Checks swing direction flow for playability."
            )

        generate_btn.click(
            fn=generate_map,
            inputs=[
                audio_input, diff_select, genre_select, bpm_input,
                song_name, song_author, onset_thresh, temp_slider,
                beam_input, sampling_check,
            ],
            outputs=[output_file, status_box],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Beat Saber Automapper Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
