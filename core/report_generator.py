"""
ReportLab PDF report generator for golf swing analysis.
Produces a multi-page professional report.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
)
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.colors import HexColor, white, black

from core.metrics_calculator import METRIC_IDEALS, metric_status
from core.video_processor import PHASE_LABELS


# ── Brand colors ──────────────────────────────────────────────────────────────
DARK_GREEN  = HexColor("#1a4a1a")
MID_GREEN   = HexColor("#2e7d32")
LIGHT_GREEN = HexColor("#e8f5e9")
GOLD        = HexColor("#c8a951")
GOOD_COLOR  = HexColor("#2e7d32")
FAIR_COLOR  = HexColor("#f57f17")
POOR_COLOR  = HexColor("#c62828")
GRAY_BG     = HexColor("#f5f5f5")
GRAY_TEXT   = HexColor("#757575")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def generate(
        self,
        output_path: str,
        annotated_frames: Dict,            # phase -> PIL Image
        phase_metrics: Dict[str, Dict],    # phase -> metrics dict
        phase_analyses: Dict[str, Dict],   # phase -> claude analysis dict (may be empty)
        synthesis: Optional[Dict],         # overall synthesis (may be None)
        video_filename: str = "",
    ):
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
            title="Golf Swing Analysis Report",
            author="Golf Swing Analyzer",
        )

        story = []
        story += self._cover_page(video_filename, synthesis)
        story.append(PageBreak())

        phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]
        for i, phase in enumerate(phase_order):
            if phase not in annotated_frames:
                continue
            story += self._phase_page(
                phase=phase,
                pil_image=annotated_frames.get(phase),
                metrics=phase_metrics.get(phase, {}),
                analysis=phase_analyses.get(phase, {}),
                phase_index=i,
                total_phases=len(phase_order),
            )
            story.append(PageBreak())

        story += self._summary_page(phase_metrics, synthesis)

        doc.build(story)

    # ── Cover page ────────────────────────────────────────────────────────
    def _cover_page(self, video_filename: str, synthesis: Optional[Dict]) -> list:
        elements = []
        elements.append(Spacer(1, 2 * cm))

        # Title
        elements.append(Paragraph("GOLF SWING", self.styles["BigTitle"]))
        elements.append(Paragraph("ANALYSIS REPORT", self.styles["BigTitleGold"]))
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=2, color=GOLD))
        elements.append(Spacer(1, 1 * cm))

        # Metadata
        now = datetime.now().strftime("%B %d, %Y  %H:%M")
        elements.append(Paragraph(f"Date: {now}", self.styles["Meta"]))
        if video_filename:
            elements.append(Paragraph(f"Video: {video_filename}", self.styles["Meta"]))
        elements.append(Spacer(1, 1.5 * cm))

        # Score badge
        if synthesis:
            score = synthesis.get("score", 0)
            rationale = synthesis.get("rationale", "")
            elements += self._score_badge(score)
            elements.append(Spacer(1, 0.8 * cm))
            if rationale:
                elements.append(Paragraph(rationale, self.styles["BodyText"]))
        else:
            elements.append(Paragraph(
                "Metrics analysis completed. Add --api-key to enable AI coaching insights.",
                self.styles["BodyText"]
            ))

        elements.append(Spacer(1, 1.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GREEN))
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(
            "Powered by Claude Vision AI  •  Golf Swing Analyzer",
            self.styles["Footer"]
        ))
        return elements

    # ── Per-phase page ────────────────────────────────────────────────────
    def _phase_page(
        self, phase: str, pil_image, metrics: Dict,
        analysis: Dict, phase_index: int, total_phases: int
    ) -> list:
        elements = []
        label = PHASE_LABELS.get(phase, phase.replace("_", " ").title())

        # Phase header bar
        header_table = Table(
            [[Paragraph(label.upper(), self.styles["PhaseHeader"]),
              Paragraph(f"Phase {phase_index + 1} of {total_phases}", self.styles["PhaseSubHeader"])]],
            colWidths=[12 * cm, 5 * cm],
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
            ("TOPPADDING",  (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (0, -1), 10),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Phase timeline
        elements.append(self._phase_timeline(phase_index, total_phases))
        elements.append(Spacer(1, 0.5 * cm))

        # Image + metrics panel side-by-side
        usable_w = PAGE_W - 2 * MARGIN
        img_w = usable_w * 0.58
        met_w = usable_w * 0.38
        gap_w = usable_w * 0.04

        # Build image cell
        img_cell = self._image_cell(pil_image, img_w)

        # Build metrics cell
        met_cell = self._metrics_panel(metrics, met_w)

        side_table = Table(
            [[img_cell, Spacer(gap_w, 1), met_cell]],
            colWidths=[img_w, gap_w, met_w],
        )
        side_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(side_table)
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=DARK_GREEN))
        elements.append(Spacer(1, 0.3 * cm))

        # Claude analysis section
        if analysis:
            elements += self._analysis_block(analysis)
        else:
            elements.append(Paragraph(
                "AI coaching analysis not available (run without --no-claude to enable).",
                self.styles["Note"]
            ))

        return elements

    # ── Summary page ──────────────────────────────────────────────────────
    def _summary_page(self, phase_metrics: Dict, synthesis: Optional[Dict]) -> list:
        elements = []

        # Header
        elements.append(Paragraph("SWING SUMMARY & RECOMMENDATIONS", self.styles["PhaseHeader2"]))
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=2, color=GOLD))
        elements.append(Spacer(1, 0.5 * cm))

        # Metrics comparison table
        elements.append(Paragraph("Biomechanical Metrics Across Phases", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(self._metrics_table(phase_metrics))
        elements.append(Spacer(1, 0.8 * cm))

        if synthesis:
            # Strengths
            strengths = synthesis.get("strengths", [])
            if strengths:
                elements.append(Paragraph("Top Strengths", self.styles["SectionHeader"]))
                for s in strengths:
                    elements.append(Paragraph(f"<font color='#2e7d32'>✓</font>  {s}", self.styles["BulletGood"]))
                elements.append(Spacer(1, 0.6 * cm))

            # Priorities
            priorities = synthesis.get("priorities", [])
            if priorities:
                elements.append(Paragraph("Priority Areas for Improvement", self.styles["SectionHeader"]))
                for i, p in enumerate(priorities, 1):
                    elements.append(Paragraph(f"<font color='#f57f17'>▶</font>  {i}. {p}", self.styles["BulletFair"]))
                elements.append(Spacer(1, 0.6 * cm))

            # Practice plan
            plan = synthesis.get("practice_plan", "")
            if plan:
                elements.append(Paragraph("2-Week Practice Plan", self.styles["SectionHeader"]))
                elements.append(Paragraph(plan, self.styles["BodyText"]))

        elements.append(Spacer(1, 1 * cm))
        elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GREEN))
        elements.append(Paragraph(
            "Generated by Golf Swing Analyzer  •  Powered by Claude Vision AI",
            self.styles["Footer"]
        ))
        return elements

    # ── Component builders ────────────────────────────────────────────────
    def _score_badge(self, score: int) -> list:
        if score >= 80:
            color = GOOD_COLOR
            label = "EXCELLENT"
        elif score >= 65:
            color = FAIR_COLOR
            label = "GOOD"
        else:
            color = POOR_COLOR
            label = "NEEDS WORK"

        badge_data = [[
            Paragraph(f"<font size=40><b>{score}</b></font>", self.styles["ScoreNumber"]),
            Paragraph(f"/100\n{label}", self.styles["ScoreLabel"]),
        ]]
        badge = Table(badge_data, colWidths=[3 * cm, 4 * cm])
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("TEXTCOLOR", (0, 0), (-1, -1), white),
            ("ROUNDEDCORNERS", [8]),
            ("TOPPADDING",    (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING",   (0, 0), (0, -1), 20),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return [badge]

    def _phase_timeline(self, current: int, total: int):
        phases = ["Address", "Backswing", "Top", "Downswing", "Impact", "Follow Through"]
        cells = []
        for i, name in enumerate(phases[:total]):
            if i == current:
                style = self.styles["TimelineActive"]
            else:
                style = self.styles["TimelineInactive"]
            cells.append(Paragraph(name, style))
        col_w = (PAGE_W - 2 * MARGIN) / total
        table = Table([cells], colWidths=[col_w] * total)
        table.setStyle(TableStyle([
            ("BACKGROUND", (current, 0), (current, 0), DARK_GREEN),
            ("TEXTCOLOR",  (current, 0), (current, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, GRAY_TEXT),
        ]))
        return table

    def _image_cell(self, pil_image, max_width: float):
        if pil_image is None:
            return Paragraph("(Frame not available)", self.styles["Note"])
        img_buf = io.BytesIO()
        pil_image.save(img_buf, format="JPEG", quality=88)
        img_buf.seek(0)
        orig_w, orig_h = pil_image.size
        aspect = orig_h / orig_w
        img_h = max_width * aspect
        max_h = 12 * cm
        if img_h > max_h:
            img_h = max_h
            max_width = img_h / aspect
        return RLImage(img_buf, width=max_width, height=img_h)

    def _metrics_panel(self, metrics: Dict, panel_width: float):
        rows = []
        for key, info in METRIC_IDEALS.items():
            val = metrics.get(key)
            label = info["label"]
            unit = info["unit"]
            ideal_str = f"{info['ideal_min']}-{info['ideal_max']}{unit}"
            if val is not None:
                val_str = f"{val:.1f}{unit}"
                status = metric_status(key, val)
            else:
                val_str = "N/A"
                status = "unknown"

            status_color = {
                "good": GOOD_COLOR, "fair": FAIR_COLOR,
                "poor": POOR_COLOR, "unknown": GRAY_TEXT,
            }.get(status, GRAY_TEXT)

            rows.append([
                Paragraph(label, self.styles["MetricLabel"]),
                Paragraph(f"<font color='{status_color.hexval()}'><b>{val_str}</b></font>",
                          self.styles["MetricValue"]),
                Paragraph(f"({ideal_str})", self.styles["MetricIdeal"]),
            ])

        col_w = [panel_width * 0.44, panel_width * 0.28, panel_width * 0.28]
        table = Table(rows, colWidths=col_w, repeatRows=0)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), GRAY_BG),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, GRAY_BG]),
            ("GRID",   (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        return table

    def _analysis_block(self, analysis: Dict) -> list:
        elements = []
        obs = analysis.get("observation", "")
        strengths = analysis.get("strengths", [])
        faults = analysis.get("faults", [])
        drill = analysis.get("drill", "")

        if obs:
            elements.append(Paragraph(
                f"<b>Coach's Eye:</b> {obs}", self.styles["CoachObs"]
            ))
            elements.append(Spacer(1, 0.2 * cm))

        col_w = (PAGE_W - 2 * MARGIN) / 2 - 0.3 * cm
        left_items, right_items = [], []

        if strengths:
            left_items.append(Paragraph("Strengths", self.styles["MiniHeader"]))
            for s in strengths:
                left_items.append(Paragraph(f"<font color='#2e7d32'>✓</font>  {s}", self.styles["BulletGood"]))

        if faults:
            right_items.append(Paragraph("Areas to Improve", self.styles["MiniHeader"]))
            for f in faults:
                right_items.append(Paragraph(f"<font color='#c62828'>▶</font>  {f}", self.styles["BulletFair"]))

        if left_items or right_items:
            two_col = Table(
                [[left_items or [""], right_items or [""]]],
                colWidths=[col_w, col_w],
            )
            two_col.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elements.append(two_col)
            elements.append(Spacer(1, 0.3 * cm))

        if drill:
            elements.append(Paragraph(
                f"<b>Practice Drill:</b> {drill}", self.styles["DrillText"]
            ))

        return elements

    def _metrics_table(self, phase_metrics: Dict) -> Table:
        phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]
        phase_labels_short = ["Address", "Backswing", "Top", "Downswing", "Impact", "Follow"]
        present_phases = [(p, l) for p, l in zip(phase_order, phase_labels_short) if p in phase_metrics]

        header = ["Metric"] + [l for _, l in present_phases]
        rows = [header]

        for key, info in METRIC_IDEALS.items():
            row = [info["label"]]
            for phase, _ in present_phases:
                val = phase_metrics.get(phase, {}).get(key)
                if val is not None:
                    status = metric_status(key, val)
                    color = {"good": "#2e7d32", "fair": "#f57f17", "poor": "#c62828"}.get(status, "#757575")
                    row.append(Paragraph(
                        f"<font color='{color}'>{val:.1f}{info['unit']}</font>",
                        self.styles["TableCell"]
                    ))
                else:
                    row.append(Paragraph("—", self.styles["TableCell"]))
            rows.append(row)

        n_cols = len(header)
        usable_w = PAGE_W - 2 * MARGIN
        label_w = usable_w * 0.22
        data_w = (usable_w - label_w) / max(1, n_cols - 1)
        col_widths = [label_w] + [data_w] * (n_cols - 1)

        table = Table(rows, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
            ("TEXTCOLOR",  (0, 0), (-1, 0), white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, GRAY_BG]),
            ("GRID",   (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        return table

    # ── Style setup ───────────────────────────────────────────────────────
    def _setup_styles(self):
        s = self.styles

        def add(name, **kwargs):
            if name not in s:
                s.add(ParagraphStyle(name=name, **kwargs))
            else:
                # update existing
                for k, v in kwargs.items():
                    setattr(s[name], k, v)

        add("BigTitle",    fontSize=32, textColor=DARK_GREEN, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceAfter=4)
        add("BigTitleGold", fontSize=28, textColor=GOLD, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceAfter=8)
        add("Meta",        fontSize=10, textColor=GRAY_TEXT, alignment=TA_CENTER,
            fontName="Helvetica", spaceAfter=4)
        add("Footer",      fontSize=8,  textColor=GRAY_TEXT, alignment=TA_CENTER,
            fontName="Helvetica-Oblique")
        add("ScoreNumber", fontSize=40, textColor=white, alignment=TA_CENTER,
            fontName="Helvetica-Bold")
        add("ScoreLabel",  fontSize=14, textColor=white, alignment=TA_LEFT,
            fontName="Helvetica-Bold")
        add("PhaseHeader", fontSize=16, textColor=white, fontName="Helvetica-Bold")
        add("PhaseHeader2", fontSize=16, textColor=DARK_GREEN, fontName="Helvetica-Bold",
            spaceAfter=4)
        add("PhaseSubHeader", fontSize=10, textColor=LIGHT_GREEN, fontName="Helvetica",
            alignment=TA_RIGHT)
        add("SectionHeader", fontSize=11, textColor=DARK_GREEN, fontName="Helvetica-Bold",
            spaceAfter=4, spaceBefore=6)
        add("MiniHeader",  fontSize=9,  textColor=DARK_GREEN, fontName="Helvetica-Bold",
            spaceAfter=3)
        add("CoachObs",    fontSize=10, fontName="Helvetica-Oblique", textColor=black,
            leading=14, spaceAfter=6)
        add("DrillText",   fontSize=10, fontName="Helvetica", textColor=HexColor("#1a237e"),
            leading=14, leftIndent=10, borderPad=4)
        add("BulletGood",  fontSize=9,  fontName="Helvetica", leading=13,
            leftIndent=10, spaceAfter=2)
        add("BulletFair",  fontSize=9,  fontName="Helvetica", leading=13,
            leftIndent=10, spaceAfter=2)
        add("Note",        fontSize=9,  fontName="Helvetica-Oblique", textColor=GRAY_TEXT)
        add("MetricLabel", fontSize=8,  fontName="Helvetica-Bold",  textColor=black)
        add("MetricValue", fontSize=9,  fontName="Helvetica-Bold",  alignment=TA_CENTER)
        add("MetricIdeal", fontSize=7,  fontName="Helvetica",       textColor=GRAY_TEXT,
            alignment=TA_CENTER)
        add("TableCell",   fontSize=8,  fontName="Helvetica",       alignment=TA_CENTER)
        add("TimelineActive",   fontSize=7, fontName="Helvetica-Bold",
            textColor=white, alignment=TA_CENTER)
        add("TimelineInactive", fontSize=7, fontName="Helvetica",
            textColor=GRAY_TEXT, alignment=TA_CENTER)
        # Override default BodyText
        s["BodyText"].fontSize = 10
        s["BodyText"].leading  = 15
        s["BodyText"].spaceAfter = 6
