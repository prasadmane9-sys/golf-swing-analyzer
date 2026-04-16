#!/usr/bin/env python3
"""
Golf Swing Analyzer
====================
Analyzes a rear-view golf swing video and generates a PDF report with
annotated phase screenshots, biomechanical metrics, and AI coaching feedback.

Usage:
    python analyze.py input_video.mp4
    python analyze.py input_video.mp4 --output report.pdf --verbose
    python analyze.py input_video.mp4 --no-claude   # skip AI analysis
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        prog="analyze",
        description="Golf swing analyzer — generates PDF report from rear-view video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to input video file (MP4, MOV, AVI)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF path (default: <video_name>_report.pdf)")
    parser.add_argument("--handedness", choices=["right", "left"], default="right",
                        help="Golfer handedness (default: right)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (falls back to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip Claude Vision analysis (faster, metrics-only report)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save annotated phase frames as JPEGs alongside the PDF")
    parser.add_argument("--fps-sample", type=int, default=10,
                        help="Frames per second to sample during phase detection (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")
    return parser.parse_args()


def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg)


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {args.video}")
    if video_path.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
        print(f"Warning: Unrecognized video extension '{video_path.suffix}'. Proceeding anyway.")

    output_path = args.output or str(video_path.parent / (video_path.stem + "_report.pdf"))

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not args.no_claude and not api_key:
        sys.exit(
            "Error: Anthropic API key not found.\n"
            "  Set ANTHROPIC_API_KEY environment variable, or use --api-key, or --no-claude."
        )

    log("", args.verbose)
    log("Golf Swing Analyzer", args.verbose)
    log("=" * 40, args.verbose)
    log(f"Input:  {args.video}", args.verbose)
    log(f"Output: {output_path}", args.verbose)
    log(f"Mode:   {'Metrics only' if args.no_claude else 'Full AI analysis'}", args.verbose)
    log("", args.verbose)

    # ── Import modules (deferred so --help is fast) ───────────────────────
    try:
        from core.pose_estimator import PoseEstimator
        from core.video_processor import VideoProcessor
        from core.metrics_calculator import MetricsCalculator
        from core.claude_analyzer import ClaudeAnalyzer
        from core.report_generator import ReportGenerator
        from utils.image_utils import draw_skeleton, annotate_phase, encode_frame_jpeg, frame_to_pil
        from core.video_processor import PHASE_LABELS
    except ImportError as e:
        sys.exit(
            f"Import error: {e}\n"
            "Make sure you have activated the virtual environment and installed dependencies:\n"
            "  source venv/bin/activate\n"
            "  pip install -r requirements.txt"
        )

    # ── Step 1: Scan video for swing phases ───────────────────────────────
    log("[1/5] Scanning video for swing phases...", args.verbose)
    with PoseEstimator() as pose:
        processor = VideoProcessor(
            pose_estimator=pose,
            fps_sample=args.fps_sample,
            handedness=args.handedness,
            verbose=args.verbose,
        )
        try:
            phase_frames = processor.scan_video(str(video_path))
        except ValueError as e:
            sys.exit(f"Error during video processing: {e}")

    # ── Step 2: Compute biomechanical metrics ─────────────────────────────
    log("\n[2/5] Computing biomechanical metrics...", args.verbose)
    calc = MetricsCalculator(handedness=args.handedness)

    # Set address baseline first
    addr_frame = phase_frames.get("address")
    if addr_frame and addr_frame.landmarks:
        calc.set_address_baseline(addr_frame.landmarks)

    phase_metrics = {}
    for phase, pf in phase_frames.items():
        if pf.landmarks:
            metrics = calc.compute_all(pf.landmarks)
        else:
            metrics = {}
        phase_metrics[phase] = metrics

        if args.verbose and metrics:
            log(f"  {phase:20s}", args.verbose)
            for k, v in metrics.items():
                if v is not None:
                    from core.metrics_calculator import METRIC_IDEALS, metric_status
                    info = METRIC_IDEALS.get(k, {})
                    label = info.get("label", k)
                    unit = info.get("unit", "")
                    status = metric_status(k, v)
                    status_icon = {"good": "OK", "fair": "FAIR", "poor": "CHECK"}.get(status, "")
                    log(f"    {label:25s}: {v:6.1f}{unit:<3}  {status_icon}", args.verbose)

    # ── Step 3: Generate annotated frames ─────────────────────────────────
    log("\n[3/5] Generating annotated frames...", args.verbose)
    annotated_pil = {}
    for phase, pf in phase_frames.items():
        label = PHASE_LABELS.get(phase, phase)
        annotated = draw_skeleton(pf.frame_bgr, pf.landmarks, pf.confidence)
        annotated = annotate_phase(annotated, label, phase_metrics.get(phase), pf.confidence)
        annotated_pil[phase] = frame_to_pil(annotated)

    # ── Step 4: Claude Vision analysis ────────────────────────────────────
    phase_analyses = {}
    synthesis = None

    if not args.no_claude:
        log("\n[4/5] Running Claude Vision analysis...", args.verbose)
        analyzer = ClaudeAnalyzer(api_key=api_key)

        # Encode annotated frames as JPEG bytes for API
        frames_jpeg = {}
        for phase, pil_img in annotated_pil.items():
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            frames_jpeg[phase] = buf.getvalue()

        phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]

        for phase in phase_order:
            if phase not in frames_jpeg:
                continue
            log(f"  Analyzing {phase}...", args.verbose)
            try:
                analysis = analyzer.analyze_phase(
                    frame_jpeg_bytes=frames_jpeg[phase],
                    phase=phase,
                    metrics=phase_metrics.get(phase, {}),
                    handedness=args.handedness,
                )
                phase_analyses[phase] = analysis
            except Exception as e:
                log(f"  Warning: Claude API error for {phase}: {e}", True)
                phase_analyses[phase] = {}

        log("  Running synthesis...", args.verbose)
        try:
            ordered_jpegs = [frames_jpeg[p] for p in phase_order if p in frames_jpeg]
            synthesis = analyzer.synthesize(
                frames_jpeg=ordered_jpegs,
                all_metrics=phase_metrics,
                phase_analyses=phase_analyses,
                handedness=args.handedness,
            )
            if args.verbose:
                log(f"  Overall score: {synthesis.get('score', '?')}/100", True)
        except Exception as e:
            log(f"  Warning: Synthesis failed: {e}", True)
            synthesis = None
    else:
        log("\n[4/5] Skipping Claude analysis (--no-claude).", args.verbose)

    # ── Step 5: Generate PDF report ───────────────────────────────────────
    log("\n[5/5] Generating PDF report...", args.verbose)
    generator = ReportGenerator()
    try:
        generator.generate(
            output_path=output_path,
            annotated_frames=annotated_pil,
            phase_metrics=phase_metrics,
            phase_analyses=phase_analyses,
            synthesis=synthesis,
            video_filename=video_path.name,
        )
    except Exception as e:
        sys.exit(f"Error generating PDF: {e}")

    # ── Save individual frames if requested ───────────────────────────────
    if args.save_frames:
        frames_dir = Path(output_path).parent / (Path(output_path).stem + "_frames")
        frames_dir.mkdir(exist_ok=True)
        for phase, pil_img in annotated_pil.items():
            pil_img.save(str(frames_dir / f"{phase}.jpg"), quality=90)
        log(f"  Frames saved to: {frames_dir}/", args.verbose)

    print(f"\nReport saved: {output_path}")
    if synthesis:
        score = synthesis.get("score", "?")
        print(f"Overall score: {score}/100")


if __name__ == "__main__":
    main()
