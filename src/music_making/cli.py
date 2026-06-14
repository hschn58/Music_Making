"""make-song CLI."""

from __future__ import annotations

import argparse
import sys

from . import storyboard
from .orchestrator import produce


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make-song",
        description="Compose an original track from a scene (text situation or video).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--situation", help="Describe the scene the song is about.")
    src.add_argument("--video", help="Path to a video whose scene drives the music.")
    src.add_argument("--images", nargs="+", metavar="IMG",
                     help="Scene photos, in story order (a literal storyboard).")
    p.add_argument("--genre", default="smooth-funk", help="Genre preset (default: smooth-funk).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=30.0, help="Target seconds (text mode).")
    p.add_argument("--seconds-per-scene", type=float, default=10.0,
                   help="Seconds each image holds (images mode).")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="out", help="Output directory.")
    p.add_argument("--soundfont", default=None, help="Override SoundFont path.")
    p.add_argument("--attempts", type=int, default=2, help="Max QC attempts.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    timbres = None
    if args.images:
        sb, timbres = storyboard.from_images(args.images, genre=args.genre, seed=args.seed,
                                             seconds_per_scene=args.seconds_per_scene,
                                             title=args.title)
    elif args.video:
        sb = storyboard.from_video(args.video, genre=args.genre, seed=args.seed, title=args.title)
    else:
        sb = storyboard.from_text(args.situation, genre=args.genre, seed=args.seed,
                                  duration_sec=args.duration, title=args.title)

    print(f"Scene: {sb.title}  |  {sb.genre}  |  {sb.tempo_bpm} bpm  |  {sb.key}  "
          f"|  {sb.duration_sec:.1f}s", file=sys.stderr)
    track = produce(sb, args.out, soundfont=args.soundfont, max_attempts=args.attempts,
                    timbres=timbres)

    qc = track.qc
    status = "PASS" if qc.passed else "FAIL"
    print(f"[QC {status}] scene corr={qc.mean_correlation:.2f}  "
          f"dominance={qc.dominance_accuracy:.0%}  LUFS={qc.integrated_lufs:.1f}  "
          f"peak={qc.true_peak_db:.1f}dB", file=sys.stderr)
    for s in qc.band_scores:
        print(f"  {s.band:>4} <- {s.layer:<16} corr={s.correlation:+.2f}", file=sys.stderr)
    for n in qc.notes:
        print(f"  note: {n}", file=sys.stderr)

    print(track.wav_path)
    return 0 if qc.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
