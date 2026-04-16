"""
Golf Swing Analyzer — Streamlit Web UI
Upload a video, enter your name, download the PDF report.
"""

import io
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="⛳",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    .stButton>button {
        background-color: #1a4a1a;
        color: white;
        border-radius: 8px;
        padding: 0.6em 2em;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #2e7d32; }
    .stDownloadButton>button {
        background-color: #c8a951;
        color: white;
        border-radius: 8px;
        padding: 0.6em 2em;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stDownloadButton>button:hover { background-color: #b8962e; }
    h1 { color: #1a4a1a; }
    .metric-box {
        background: white;
        border-left: 4px solid #1a4a1a;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⛳ Golf Swing Analyzer")
st.markdown("Upload your swing video, get a professional PDF report with phase screenshots and AI coaching feedback.")
st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    golfer_name = st.text_input("Golfer Name", placeholder="e.g. Tiger Woods")

with col2:
    handedness = st.selectbox("Handedness", ["Right", "Left"])

api_key = st.text_input(
    "Anthropic API Key (optional)",
    type="password",
    placeholder="sk-ant-... (leave blank for metrics-only report)",
    help="Add your Anthropic API key to get AI coaching feedback. Get one at console.anthropic.com"
)

video_file = st.file_uploader(
    "Upload Swing Video",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    help="Film from behind the golfer, facing the target. Full body should be visible.",
)

st.divider()

# ── Analyze button ────────────────────────────────────────────────────────────
analyze_clicked = st.button("Analyze Swing", disabled=(video_file is None))

if video_file is None:
    st.info("Upload a rear-view golf swing video to get started.")

# ── Analysis pipeline ─────────────────────────────────────────────────────────
if analyze_clicked and video_file is not None:

    # Resolve API key
    resolved_key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    use_claude = bool(resolved_key)

    # Save uploaded video to a temp file
    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        tmp_video_path = tmp.name

    output_pdf_path = tmp_video_path.replace(suffix, "_report.pdf")

    try:
        # ── Imports ───────────────────────────────────────────────────────
        from core.pose_estimator import PoseEstimator
        from core.video_processor import VideoProcessor, PHASE_LABELS
        from core.metrics_calculator import MetricsCalculator, METRIC_IDEALS, metric_status
        from core.claude_analyzer import ClaudeAnalyzer
        from core.report_generator import ReportGenerator
        from utils.image_utils import draw_skeleton, annotate_phase, frame_to_pil

        # ── Progress UI ───────────────────────────────────────────────────
        progress = st.progress(0, text="Starting analysis...")
        status_box = st.empty()

        # Step 1: Phase detection
        status_box.info("Step 1/5 — Detecting swing phases with pose estimation...")
        with PoseEstimator() as pose:
            processor = VideoProcessor(
                pose_estimator=pose,
                fps_sample=10,
                handedness=handedness.lower(),
                verbose=False,
            )
            phase_frames = processor.scan_video(tmp_video_path)
        progress.progress(20, text="Phases detected")

        # Step 2: Metrics
        status_box.info("Step 2/5 — Computing biomechanical metrics...")
        calc = MetricsCalculator(handedness=handedness.lower())
        addr = phase_frames.get("address")
        if addr and addr.landmarks:
            calc.set_address_baseline(addr.landmarks)

        phase_metrics = {}
        for phase, pf in phase_frames.items():
            phase_metrics[phase] = calc.compute_all(pf.landmarks) if pf.landmarks else {}
        progress.progress(40, text="Metrics computed")

        # Step 3: Annotate frames
        status_box.info("Step 3/5 — Generating annotated frames...")
        annotated_pil = {}
        for phase, pf in phase_frames.items():
            label = PHASE_LABELS.get(phase, phase)
            frame = draw_skeleton(pf.frame_bgr, pf.landmarks, pf.confidence)
            frame = annotate_phase(frame, label, phase_metrics.get(phase), pf.confidence)
            annotated_pil[phase] = frame_to_pil(frame)
        progress.progress(55, text="Frames annotated")

        # Step 4: Claude analysis
        phase_analyses = {}
        synthesis = None

        if use_claude:
            status_box.info("Step 4/5 — Running Claude AI coaching analysis...")
            analyzer = ClaudeAnalyzer(api_key=resolved_key)
            phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]

            frames_jpeg = {}
            for phase, pil_img in annotated_pil.items():
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                frames_jpeg[phase] = buf.getvalue()

            for i, phase in enumerate(phase_order):
                if phase not in frames_jpeg:
                    continue
                try:
                    phase_analyses[phase] = analyzer.analyze_phase(
                        frame_jpeg_bytes=frames_jpeg[phase],
                        phase=phase,
                        metrics=phase_metrics.get(phase, {}),
                        handedness=handedness.lower(),
                    )
                except Exception as e:
                    phase_analyses[phase] = {}
                progress.progress(55 + int((i + 1) / len(phase_order) * 25), text=f"Analyzing {phase}...")

            try:
                ordered_jpegs = [frames_jpeg[p] for p in phase_order if p in frames_jpeg]
                synthesis = analyzer.synthesize(
                    frames_jpeg=ordered_jpegs,
                    all_metrics=phase_metrics,
                    phase_analyses=phase_analyses,
                    handedness=handedness.lower(),
                )
            except Exception:
                synthesis = None
        else:
            status_box.info("Step 4/5 — Skipping AI analysis (no API key provided)...")
        progress.progress(80, text="AI analysis done")

        # Step 5: Generate PDF
        status_box.info("Step 5/5 — Generating PDF report...")
        generator = ReportGenerator()

        # Inject golfer name into report filename and cover
        report_name = golfer_name.strip() if golfer_name.strip() else "Golfer"
        video_display_name = f"{report_name} — {video_file.name}"

        generator.generate(
            output_path=output_pdf_path,
            annotated_frames=annotated_pil,
            phase_metrics=phase_metrics,
            phase_analyses=phase_analyses,
            synthesis=synthesis,
            video_filename=video_display_name,
        )
        progress.progress(100, text="Done!")
        status_box.empty()

        # ── Results ───────────────────────────────────────────────────────
        st.success("Analysis complete!")

        # Score badge
        if synthesis:
            score = synthesis.get("score", 0)
            color = "#2e7d32" if score >= 80 else ("#f57f17" if score >= 65 else "#c62828")
            label = "Excellent" if score >= 80 else ("Good" if score >= 65 else "Needs Work")
            st.markdown(f"""
            <div style="background:{color};color:white;border-radius:12px;padding:16px 24px;
                        text-align:center;margin:12px 0;">
                <span style="font-size:2.5rem;font-weight:bold;">{score}</span>
                <span style="font-size:1.1rem;">/100 — {label}</span>
            </div>
            """, unsafe_allow_html=True)

            rationale = synthesis.get("rationale", "")
            if rationale:
                st.markdown(f"*{rationale}*")

        # Key metrics at impact
        impact_metrics = phase_metrics.get("impact", {})
        if impact_metrics:
            st.markdown("**Key metrics at impact:**")
            cols = st.columns(3)
            metric_keys = ["spine_angle", "hip_rotation", "lead_arm_angle"]
            for i, key in enumerate(metric_keys):
                val = impact_metrics.get(key)
                if val is not None:
                    info = METRIC_IDEALS[key]
                    status = metric_status(key, val)
                    icon = "✅" if status == "good" else ("⚠️" if status == "fair" else "❌")
                    cols[i].metric(
                        label=f"{icon} {info['label']}",
                        value=f"{val:.1f}{info['unit']}",
                        delta=f"Ideal: {info['ideal_min']}-{info['ideal_max']}{info['unit']}",
                        delta_color="off",
                    )

        # Phase thumbnails
        st.markdown("**Detected swing phases:**")
        phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]
        thumb_cols = st.columns(6)
        for i, phase in enumerate(phase_order):
            if phase in annotated_pil:
                thumb_cols[i].image(
                    annotated_pil[phase],
                    caption=PHASE_LABELS.get(phase, phase),
                    use_container_width=True,
                )

        st.divider()

        # Download button
        with open(output_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        safe_name = report_name.replace(" ", "_")
        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=f"{safe_name}_swing_report.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        # Clean up temp files
        try:
            os.unlink(tmp_video_path)
        except Exception:
            pass

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9e9e9e;font-size:0.8rem;'>"
    "Golf Swing Analyzer · Powered by MediaPipe + Claude Vision AI"
    "</div>",
    unsafe_allow_html=True,
)
