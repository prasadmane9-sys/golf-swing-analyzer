"""
Claude Vision API integration for golf swing analysis.
Sends annotated frames + computed metrics; returns structured coaching feedback.
"""

import base64
import json
import re
from typing import Dict, List, Optional

from core.metrics_calculator import METRIC_IDEALS


SYSTEM_PROMPT = """You are a PGA-certified golf swing coach with 20+ years of experience \
analyzing swings from video footage for serious amateur golfers. Your coaching philosophy \
integrates the foundational teachings of Ben Hogan's "Five Lessons: The Modern Fundamentals \
of Golf", the Stack & Tilt methodology, and Jim McLean's X-Factor (the differential between \
shoulder and hip rotation, ideally 45-55° at the top of the backswing). You provide specific, \
actionable, and constructive feedback grounded in what is visually apparent in each image, \
always cross-referencing computed biomechanical metrics with your visual observations.

BIOMECHANICAL BENCHMARKS (Tour-level reference):
- Shoulder turn at top of backswing: 90° minimum (pros: 95-110°)
- Hip turn at top of backswing: 25-45° (pros: 30-40°); amateurs frequently over-rotate hips, \
  reducing X-Factor and power
- X-Factor (shoulder turn minus hip turn): 45-55° optimal
- Hip rotation at impact: 40-50° open to target line
- Forward shaft lean at impact: hands 2-4 inches ahead of ball (irons)
- Weight distribution at impact: 60-70% on lead foot
- Spine angle at address: 20-30° forward tilt from vertical; maintained through impact

COMMON AMATEUR FAULT DATABASE:
1. Over-the-top downswing — club exits to the left of the target line at the start of the \
   downswing (outside-in path), causing pull shots or a slice; root cause is upper-body \
   initiating the downswing before the hips clear.
2. Early extension (goat humping) — hips thrust toward the ball through impact instead of \
   rotating; eliminates room for the arms to deliver the club, often paired with a standing-up \
   move that increases spine angle, causing fat/thin contact and a blocked or hooked ball flight.
3. Sway — lateral hip slide away from the target in the backswing instead of rotating around \
   the trail hip; shifts weight incorrectly and destroys coil, leading to inconsistent contact.
4. Reverse pivot — weight stays on or shifts to the lead foot during the backswing; loses \
   power and causes topped or thin shots at impact.
5. Chicken wing — lead arm collapses at the elbow through impact and into the follow-through; \
   causes the club face to flip and produces inconsistent low draws or smothered shots.
6. Casting / early release — lag angle lost early in the downswing (wrist angles unhinge before \
   impact zone); eliminates forward shaft lean and stored power, leading to weak, high-flying shots.
7. Cupped lead wrist at top — lead wrist bent backward (cupped) at the top of the backswing, \
   opening the club face and requiring compensations through impact to square it.
8. Bowed lead wrist at top — lead wrist arched forward (bowed) at the top; closes the club face \
   and can lead to a strong draw or hook if not managed through impact.

Always cross-reference computed metrics with what you visually observe. If a metric contradicts \
what you see (e.g., pose-estimation error), note the discrepancy and trust the visual evidence. \
Give feedback calibrated for serious amateur golfers who understand golf terminology and want \
precise, technical coaching — not generic tips."""

PHASE_DESCRIPTIONS = {
    "address":          "the setup/address position (golfer at rest, about to start the swing)",
    "backswing":        "the mid-backswing position",
    "top_of_backswing": "the top of the backswing (transition point)",
    "downswing":        "the early downswing / transition",
    "impact":           "the impact position (club meeting the ball)",
    "follow_through":   "the follow-through / finish position",
}

# Ideal reference values per phase (to contextualize metrics in the prompt)
PHASE_IDEALS = {
    "address":          {"spine_angle": "20-30°", "hip_rotation": "0°",    "lead_arm_angle": "155-170°"},
    "backswing":        {"spine_angle": "20-30°", "hip_rotation": "15-25°", "lead_arm_angle": "155-175°"},
    "top_of_backswing": {"spine_angle": "20-30°", "hip_rotation": "25-45°", "lead_arm_angle": "165-180°"},
    "downswing":        {"spine_angle": "20-30°", "hip_rotation": "30-45°", "lead_arm_angle": "165-180°"},
    "impact":           {"spine_angle": "20-35°", "hip_rotation": "40-50°", "lead_arm_angle": "170-180°"},
    "follow_through":   {"spine_angle": "upright", "hip_rotation": "45-65°", "lead_arm_angle": "any"},
}

# Per-phase coaching checklists injected into each phase prompt
PHASE_CHECKLISTS = {
    "address": """\
Address checklist to evaluate:
- Stance width: approximately shoulder-width for irons (wider for driver)
- Ball position: center for short irons, progressive forward through the bag to inside lead heel for driver
- Spine tilt: slight tilt away from target (trail shoulder lower than lead), forward tilt 20-30° from vertical
- Knee flex: 15-20° of flex (athletic posture, not squatting)
- Grip pressure: light to moderate (7/10 scale max); no tension visible in forearms
- Weight distribution: 50/50 between feet (neutral), slightly favoring trail foot for driver
- Arms hanging naturally under the shoulders; lead arm relatively straight""",

    "backswing": """\
Backswing checklist to evaluate:
- One-piece takeaway: club, hands, arms, and chest move together for the first 18-24 inches
- Club stays outside the hands during early takeaway (no early inside roll)
- Lead arm stays connected to the chest; minimal gap opening between lead arm and chest
- No early wrist hinge (wrist set should begin when hands reach hip height or slightly above)
- Hips resist rotation early (X-Factor begins building) — look for trail hip staying behind lead hip
- Weight beginning to shift to inside of trail foot; no lateral sway of trail hip""",

    "top_of_backswing": """\
Top of backswing checklist to evaluate:
- Lead arm at or slightly above parallel to the ground (a laid-off or across-the-line club position is a fault)
- Shoulder turn: minimum 90° from address (ideally 95-110°); use chest/upper-back rotation as reference
- Wrist hinge: approximately 90° wrist cock — club shaft forms ~90° angle with lead forearm
- Hip turn: should not exceed 45°; excessive hip rotation reduces X-Factor coil
- X-Factor: shoulder turn minus hip turn should be 45-55° for a power source
- Weight loaded onto inside of trail foot; lead heel may rise slightly (acceptable) but no sway
- Spine angle maintained from address — no standing up or increasing forward tilt
- Lead wrist position: flat to slightly bowed is ideal; cupped wrist opens the face""",

    "downswing": """\
Downswing checklist to evaluate:
- Lower body initiates: hips begin clearing toward target BEFORE shoulders and arms drop (sequencing)
- Lag maintained: the wrist hinge angle (L-shape between lead forearm and club shaft) is preserved deep into downswing
- Club drops into the slot: club shaft should shallow and approach from inside the target line (not over-the-top)
- Head stays behind the ball position at address — no forward head drift
- Trail elbow tucks toward trail hip (not flying out)
- Weight shifting aggressively to lead side by mid-downswing
- Hips beginning to clear (rotate open) while shoulders remain relatively closed — X-Factor stretch""",

    "impact": """\
Impact checklist to evaluate:
- Hips open to target: 40-50° of hip rotation (hips clearly ahead of the club and shoulders)
- Forward shaft lean: hands visibly ahead of the ball (irons); shaft leaning toward target
- Weight distribution: 60-70% on lead foot (lead knee driving toward target)
- Head behind the ball: head should be at or slightly behind ball position at address
- Lead arm straight (or nearly so): no chicken wing; lead elbow points toward lead hip
- Square club face relative to swing path at the moment of contact
- Trail heel beginning to lift as weight shifts fully to lead side""",

    "follow_through": """\
Follow-through checklist to evaluate:
- Full hip rotation to finish: belt buckle facing the target (or just left of target for right-handers)
- Trail foot up on the toe: only the toe of the trail foot should remain on the ground
- Balanced finish on lead side: all weight on lead foot, lead leg straight and stable
- High hands: hands finish high (above left shoulder for right-handers) indicating a full, free release
- Chest faces target or slightly past target
- No loss of balance or falling backward — weight fully transferred""",
}


class ClaudeAnalyzer:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_phase(
        self,
        frame_jpeg_bytes: bytes,
        phase: str,
        metrics: Dict,
        handedness: str = "right",
    ) -> Dict:
        """
        Analyze a single swing phase frame.
        Returns dict with keys: observation, strengths, faults, drill.
        """
        b64 = base64.standard_b64encode(frame_jpeg_bytes).decode("utf-8")
        prompt = self._build_phase_prompt(phase, metrics, handedness)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = message.content[0].text
        return self._parse_phase_response(raw, phase)

    def synthesize(
        self,
        frames_jpeg: List[bytes],
        all_metrics: Dict[str, Dict],
        phase_analyses: Dict[str, Dict],
        handedness: str = "right",
    ) -> Dict:
        """
        Send all 6 annotated frames + per-phase summaries for an overall assessment.
        Returns dict with keys: score, rationale, strengths, priorities, practice_plan.
        """
        content = []
        phase_order = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]
        for i, phase in enumerate(phase_order):
            if i < len(frames_jpeg):
                b64 = base64.standard_b64encode(frames_jpeg[i]).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })

        prompt = self._build_synthesis_prompt(all_metrics, phase_analyses, handedness)
        content.append({"type": "text", "text": prompt})

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": content}],
        )
        raw = message.content[0].text
        return self._parse_synthesis_response(raw)

    # ── Prompt builders ───────────────────────────────────────────────────
    def _build_phase_prompt(self, phase: str, metrics: Dict, handedness: str) -> str:
        phase_desc = PHASE_DESCRIPTIONS.get(phase, phase)
        ideals = PHASE_IDEALS.get(phase, {})
        checklist = PHASE_CHECKLISTS.get(phase, "")

        metrics_lines = []
        for key, info in METRIC_IDEALS.items():
            val = metrics.get(key)
            ideal_str = ideals.get(key, f"{info['ideal_min']}-{info['ideal_max']}{info['unit']}")
            if val is not None:
                metrics_lines.append(
                    f"- {info['label']}: {val:.1f}{info['unit']}  (ideal: {ideal_str})"
                )
            else:
                metrics_lines.append(
                    f"- {info['label']}: not detected  (ideal: {ideal_str})"
                )

        metrics_block = "\n".join(metrics_lines) if metrics_lines else "  (metrics unavailable)"
        hand_note = "right" if handedness == "right" else "left"

        return f"""This image shows a {hand_note}-handed golfer at {phase_desc}, filmed from behind \
(rear view, target to the left).

Computed biomechanical metrics (cross-reference with visual evidence; flag any discrepancies):
{metrics_block}

{checklist}

Please provide your analysis in EXACTLY this format:

OBSERVATION: [1-2 sentences — the single most important thing you notice visually at this phase, \
referencing specific body segments or club positions]

STRENGTHS:
• [strength 1 — be specific, e.g., "Lead arm maintains connection to chest through mid-backswing"]
• [strength 2 — omit bullet entirely if no second strength]

FAULTS:
• [fault 1 — name the fault precisely, e.g., "Early extension: hips thrusting toward the ball"]
• [fault 2 — omit bullet entirely if no second fault]

DRILL: [1 specific, named practice drill to address the most critical fault — include setup, \
key feel, and rep scheme. Use "Maintain current form and focus on rhythm" if no faults detected.]

Total response: under 220 words."""

    def _build_synthesis_prompt(
        self, all_metrics: Dict, phase_analyses: Dict, handedness: str
    ) -> str:
        summaries = []
        for phase in ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]:
            analysis = phase_analyses.get(phase, {})
            obs = analysis.get("observation", "N/A")
            faults = "; ".join(analysis.get("faults", [])) or "none noted"
            summaries.append(f"  {phase}: {obs}  Faults: {faults}")
        summaries_block = "\n".join(summaries)

        impact = all_metrics.get("impact", {})
        impact_lines = []
        for key, info in METRIC_IDEALS.items():
            val = impact.get(key)
            if val is not None:
                impact_lines.append(f"  {info['label']}: {val:.1f}{info['unit']}")
            else:
                impact_lines.append(f"  {info['label']}: not detected")
        impact_block = "\n".join(impact_lines) or "  (unavailable)"

        return f"""You have seen all 6 frames of this {"right" if handedness == "right" else "left"}-handed \
golfer's swing (address through follow-through), filmed from behind. You have also reviewed the \
per-phase analyses below. Now synthesize a complete, holistic assessment.

Per-phase summaries:
{summaries_block}

Key impact metrics:
{impact_block}

SCORING RUBRIC (handicap-aligned):
- 85-100: Scratch to +5 level mechanics — tour-caliber positions, minimal compensations
- 70-84: 5-15 handicap — solid fundamentals with one or two minor mechanical flaws
- 55-69: 15-25 handicap — clear mechanical issues in 2-3 phases that require structured practice
- 40-54: 25+ handicap — multiple fundamental issues affecting consistency and power
- Below 40: Beginner — foundational movement patterns need to be established before refinement

Apply the scoring rubric honestly. Cross-reference the computed metrics with your visual observations \
of all 6 frames. Identify patterns that recur across multiple phases (e.g., early extension showing \
up in both downswing and impact). Prioritize the faults that most impact ball striking and scoring.

Respond in valid JSON (no markdown fences) with exactly these keys:
{{
  "score": <integer 0-100>,
  "rationale": "<2-3 sentences explaining the score, referencing the handicap band and the 1-2 most \
impactful mechanical patterns observed across the full swing>",
  "strengths": ["<strength 1 — specific, referencing phase and body segment>", \
"<strength 2>", "<strength 3>"],
  "priorities": ["<priority 1 — most impactful fault to fix, with phase reference>", \
"<priority 2>", "<priority 3>"],
  "practice_plan": "<3-4 sentence 2-week practice plan: specify drills by name, rep counts, and \
which session types (range vs. short game) to prioritize. Tie each drill to a specific fault from \
the priorities list.>"
}}"""

    # ── Response parsers ──────────────────────────────────────────────────
    def _parse_phase_response(self, raw: str, phase: str) -> Dict:
        result = {
            "observation": "",
            "strengths": [],
            "faults": [],
            "drill": "",
            "raw": raw,
        }
        try:
            obs_match = re.search(r"OBSERVATION:\s*(.+?)(?=\n|STRENGTHS|$)", raw, re.IGNORECASE | re.DOTALL)
            if obs_match:
                result["observation"] = obs_match.group(1).strip()

            strengths_match = re.search(r"STRENGTHS:(.*?)(?=FAULTS:|DRILL:|$)", raw, re.IGNORECASE | re.DOTALL)
            if strengths_match:
                result["strengths"] = _extract_bullets(strengths_match.group(1))

            faults_match = re.search(r"FAULTS:(.*?)(?=DRILL:|$)", raw, re.IGNORECASE | re.DOTALL)
            if faults_match:
                result["faults"] = _extract_bullets(faults_match.group(1))

            drill_match = re.search(r"DRILL:\s*(.+?)$", raw, re.IGNORECASE | re.DOTALL)
            if drill_match:
                result["drill"] = drill_match.group(1).strip()

        except Exception:
            # Parsing failure: keep raw text as observation
            result["observation"] = raw[:300]
        return result

    def _parse_synthesis_response(self, raw: str) -> Dict:
        default = {
            "score": 70,
            "rationale": "Analysis completed.",
            "strengths": [],
            "priorities": [],
            "practice_plan": "",
        }
        try:
            # Strip potential markdown fences
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(clean)
            default.update(data)
        except json.JSONDecodeError:
            # Fallback: try to extract score with regex
            score_match = re.search(r'"score"\s*:\s*(\d+)', raw)
            if score_match:
                default["score"] = int(score_match.group(1))
            default["rationale"] = raw[:400]
        return default


def _extract_bullets(text: str) -> List[str]:
    """Extract bullet-point items (•, -, *) from a block of text."""
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("•", "-", "*")) and len(line) > 2:
            item = line.lstrip("•-* ").strip()
            if item and item.lower() not in ("omit if none", "none"):
                items.append(item)
    return items
