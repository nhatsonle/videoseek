import json
from typing import Any, Dict, List, Optional

from videoseek.utils import call_llm_api


VALID_QUERY_TYPES = {
    "global_narrative",
    "explicit_temporal",
    "implicit_fine_grained",
    "anomaly_ocr",
}


def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no", ""):
            return False
    if isinstance(val, (int, float)):
        return bool(val)
    return default


class QueryProfile:
    """
    Represents the analyzed profile of a user query for adaptive video perception.
    """

    def __init__(
        self,
        query_type: str = "global_narrative",
        recommended_fps: float = 1,
        enable_anomaly_scan: bool = False,
        target_cues: Optional[List[str]] = None,
        planning_hint: str = "",
    ):
        self.query_type = (
            query_type if query_type in VALID_QUERY_TYPES else "global_narrative"
        )
        self.recommended_fps = recommended_fps
        self.enable_anomaly_scan = _to_bool(enable_anomaly_scan, default=False)
        self.target_cues = list(target_cues) if target_cues is not None else []
        self.planning_hint = str(planning_hint) if planning_hint is not None else ""

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "recommended_fps": self.recommended_fps,
            "enable_anomaly_scan": self.enable_anomaly_scan,
            "target_cues": self.target_cues,
            "planning_hint": self.planning_hint,
        }


QUERY_ROUTER_SYSTEM_PROMPT = """You are an expert Query Router for a Long-Form Video Understanding Agent.
Your task is to analyze the user's question and classify it into ONE of four `query_type` categories to determine the optimal visual sampling rate (FPS) and anomaly scanning strategy:

1. "global_narrative":
   - Global questions about overall video theme, summary, main storyline, or cause-and-effect across the whole video.
   - Default parameters: `recommended_fps` = 1, `enable_anomaly_scan` = false.

2. "explicit_temporal":
   - Questions that explicitly mention a specific timestamp or time range (e.g., "at 02:15", "between 100s and 120s").
   - Default parameters: `recommended_fps` = 4, `enable_anomaly_scan` = false.

3. "implicit_fine_grained":
   - Questions asking about fine-grained actions, small objects, gestures, or specific visual details in a particular scene without an explicit timestamp.
   - Default parameters: `recommended_fps` = 4, `enable_anomaly_scan` = false.

4. "anomaly_ocr":
   - Questions asking about written text/OCR, signs, numbers, subtitles/captions embedded in scenes, distinct colors, or brief/sudden/unexpected non-narrative events (flashing text, abrupt visual anomalies).
   - Default parameters: `recommended_fps` = 8, `enable_anomaly_scan` = true.

Extract `target_cues` (a list of 1 to 5 key visual objects, text strings, colors, or event keywords to watch out for) and a concise `planning_hint` (1-2 sentences advising the agent on how to use `overview`, `skim`, and `focus` with the recommended `fps`).

Return ONLY valid JSON matching this exact schema:
{
  "query_type": "anomaly_ocr",
  "recommended_fps": 8,
  "enable_anomaly_scan": true,
  "target_cues": ["keyword1", "keyword2"],
  "planning_hint": "Concise strategy hint for the video agent."
}
"""


class QueryAnalyzer:
    """
    Analyzes a video QA question using a single LLM JSON Router call.
    """

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def default_profile() -> QueryProfile:
        return QueryProfile(
            query_type="global_narrative",
            recommended_fps=1,
            enable_anomaly_scan=False,
            target_cues=[],
            planning_hint="",
        )

    def analyze(self, question: str) -> QueryProfile:
        if not question or not str(question).strip():
            return self.default_profile()

        try:
            response = call_llm_api(
                messages=[
                    {"role": "system", "content": QUERY_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question:\n{question}"},
                ],
                model_name=self.config.get("model_name", "gpt-5"),
                api_base=self.config.get("api_base"),
                api_key=self.config.get("api_key"),
                api_version=self.config.get("api_version"),
                max_tokens=1024,
                reasoning_effort="low",
                seed=self.config.get("seed", 42),
                temperature=self.config.get("temperature", 1.0),
                return_json=True,
            )
            if response is None:
                return self.default_profile()

            content = response.choices[0].message.content
            data = json.loads(content)
            if not isinstance(data, dict):
                return self.default_profile()

            query_type = data.get("query_type", "global_narrative")
            if query_type not in VALID_QUERY_TYPES:
                query_type = "global_narrative"

            default_fps_map = {
                "global_narrative": 1,
                "explicit_temporal": 4,
                "implicit_fine_grained": 4,
                "anomaly_ocr": 8,
            }
            recommended_fps = data.get(
                "recommended_fps", default_fps_map[query_type]
            )
            try:
                recommended_fps = float(recommended_fps)
                if recommended_fps <= 0:
                    recommended_fps = default_fps_map[query_type]
            except (TypeError, ValueError):
                recommended_fps = default_fps_map[query_type]

            enable_anomaly_scan = _to_bool(
                data.get("enable_anomaly_scan", query_type == "anomaly_ocr"),
                default=(query_type == "anomaly_ocr"),
            )
            target_cues = data.get("target_cues", [])
            if not isinstance(target_cues, list):
                target_cues = [str(target_cues)]
            else:
                target_cues = [str(c) for c in target_cues if c is not None]

            planning_hint = data.get("planning_hint", "")

            return QueryProfile(
                query_type=query_type,
                recommended_fps=recommended_fps,
                enable_anomaly_scan=enable_anomaly_scan,
                target_cues=target_cues,
                planning_hint=str(planning_hint),
            )
        except Exception as e:
            print(f"QueryAnalyzer fallback to default profile due to error: {e}")
            return self.default_profile()
