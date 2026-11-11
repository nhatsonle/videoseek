import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure test suite can run even in lightweight environments where decord/litellm are not installed
if "decord" not in sys.modules:
    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = MagicMock()
    sys.modules["decord"] = fake_decord

if "litellm" not in sys.modules:
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = MagicMock()
    sys.modules["litellm"] = fake_litellm

from config import general_config, prompts_config
from videoseek.agent import ReflexionEvaluatorAgent, VideoSeekAgent
from videoseek.tools.inspect import execute_inspect, inspect_tool


class DummyBatch:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


class FakeVideoReader:
    def __init__(self, total_frames: int = 3600, fps: float = 30.0):
        self.total_frames = total_frames
        self.fps = fps

    def __len__(self) -> int:
        return self.total_frames

    def get_avg_fps(self) -> float:
        return self.fps

    def get_batch(self, indices) -> DummyBatch:
        frames = [
            np.full((64, 64, 3), int(idx) % 255, dtype=np.uint8)
            for idx in indices
        ]
        return DummyBatch(np.stack(frames, axis=0))


def _make_llm_response(content: str = "", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "content": content,
        "tool_calls": tool_calls,
    }
    msg.json.return_value = json.dumps(
        {"content": content, "tool_calls": tool_calls}
    )
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


class TestUnifiedInspectAndReflexion(unittest.TestCase):
    def setUp(self):
        self.config = {
            **general_config,
            **prompts_config,
            "model_name": "gpt-6.1-terra",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "api_version": None,
            "max_inspect_frames": 64,
            "reflexion_enabled": True,
        }

    def test_1_inspect_tool_schema(self):
        """Test 1: Verify strict schema and required parameters of `inspect_tool`."""
        self.assertEqual(inspect_tool["function"]["name"], "inspect")
        self.assertTrue(inspect_tool["function"]["strict"])
        params = inspect_tool["function"]["parameters"]
        self.assertEqual(
            params["required"], ["query", "start_time", "end_time", "fps"]
        )
        for key in ["query", "start_time", "end_time", "fps"]:
            self.assertIn(key, params["properties"])
        self.assertFalse(params["additionalProperties"])

    @patch("videoseek.tools.inspect.call_llm_api")
    def test_2_execute_inspect_macro_grid_mode(self, mock_llm):
        """Test 2: Macro Grid Mode (N > 16) packs 8 frames per 2x4 grid image with Timestamp Matrix."""
        mock_llm.return_value = _make_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "0.0s",
                            "description": "Opening scene of the video.",
                        },
                        {
                            "timestamp": "44.0s",
                            "description": "Candidate red flash visible on screen.",
                        },
                    ]
                }
            )
        )
        vr = FakeVideoReader(total_frames=3600, fps=30.0)  # 120s video
        # 120s * 0.5 fps = 60 frames -> rounded up to 64 (8 grids of 2x4)
        out = execute_inspect(
            self.config,
            {
                "query": "Whole-video macro scan for red flashing text",
                "start_time": 0,
                "end_time": 120,
                "fps": 0.5,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertIn("0.0s: Opening scene", out)
        self.assertIn("44.0s: Candidate red flash", out)
        self.assertTrue(mock_llm.call_args.kwargs.get("return_json"))

        sent_content = mock_llm.call_args.kwargs["messages"][0]["content"]
        image_parts = [p for p in sent_content if p["type"] == "image_url"]
        text_parts = [p["text"] for p in sent_content if p["type"] == "text"]
        # 64 frames / 8 frames per 2x4 grid = 8 grid images
        self.assertEqual(len(image_parts), 8)
        self.assertTrue(any("Timestamp Matrix:" in t for t in text_parts))

    @patch("videoseek.tools.inspect.call_llm_api")
    def test_3_execute_inspect_micro_highres_mode(self, mock_llm):
        """Test 3: Micro High-Res Mode (N <= 16) sends individual frames and sub-second .2f timestamps when fps >= 4."""
        mock_llm.return_value = _make_llm_response(
            content="At 44.50s, the red text reads 'BI MAT'."
        )
        vr = FakeVideoReader(total_frames=3600, fps=30.0)
        # 2.0s * 8 fps = 16 frames (<= 16 -> Micro High-Res Mode)
        out = execute_inspect(
            self.config,
            {
                "query": "Read the exact red text",
                "start_time": 44.0,
                "end_time": 46.0,
                "fps": 8,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertIn("BI MAT", out)
        sent_content = mock_llm.call_args.kwargs["messages"][0]["content"]
        image_parts = [p for p in sent_content if p["type"] == "image_url"]
        ts_texts = [p["text"] for p in sent_content if p["type"] == "text"][1:-1]
        self.assertEqual(len(image_parts), 16)
        self.assertEqual(len(ts_texts), 16)
        for ts in ts_texts:
            self.assertEqual(len(ts[:-1].split(".")[1]), 2)

    @patch("videoseek.tools.inspect.call_llm_api")
    def test_4_frame_budget_guardrail_clamping(self, mock_llm):
        """Test 4: Frame Budget Guardrail automatically clamps desired_frames > max_inspect_frames."""
        mock_llm.return_value = _make_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "0.0s",
                            "description": "Clamped macro scan.",
                        }
                    ]
                }
            )
        )
        vr = FakeVideoReader(total_frames=3600, fps=30.0)
        # Request 120s * 4 fps = 480 frames (> max_inspect_frames=64)
        out = execute_inspect(
            self.config,
            {
                "query": "Over-budget scan",
                "start_time": 0,
                "end_time": 120,
                "fps": 4,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertIn(
            "[System Auto-Adjustment: Clamped from 480 to 64 frames to respect frame budget]",
            out,
        )
        sent_content = mock_llm.call_args.kwargs["messages"][0]["content"]
        image_parts = [p for p in sent_content if p["type"] == "image_url"]
        # 64 frames in 2x4 grids = 8 grid images
        self.assertEqual(len(image_parts), 8)

    @patch("videoseek.agent.VideoReader")
    def test_5_dual_agent_system_prompt_separation(self, mock_vr_cls):
        """Test 5: Verify Actor Agent and Reflexion Evaluator Agent have distinct System Prompts and isolated contexts."""
        mock_vr_cls.return_value = FakeVideoReader(total_frames=3600, fps=30.0)

        actor = VideoSeekAgent(
            config=self.config,
            video_path="dummy.mp4",
            subtitle_path="dummy.srt",
            output_dir="./outputs",
            tools=["inspect"],
            verbose=False,
        )
        self.assertIsNotNone(actor.evaluator)
        actor_sys_prompt = actor.messages[0]["content"]
        eval_sys_prompt = actor.evaluator.system_prompt

        self.assertNotEqual(actor_sys_prompt, eval_sys_prompt)
        self.assertIn("Actor Agent", actor_sys_prompt)
        self.assertIn("## `inspect`: unified multi-granular video inspection", actor_sys_prompt)
        self.assertIn("independent Reflexion Evaluator Agent", eval_sys_prompt)
        self.assertIn("Sufficiency Status", eval_sys_prompt)

        # Also verify Legacy 3-tools Baseline uses LEGACY_SYSTEM_PROMPT
        baseline_cfg = {**self.config, "reflexion_enabled": False}
        baseline_actor = VideoSeekAgent(
            config=baseline_cfg,
            video_path="dummy.mp4",
            subtitle_path="dummy.srt",
            output_dir="./outputs",
            tools=["overview", "skim", "focus"],
            verbose=False,
        )
        self.assertIsNone(baseline_actor.evaluator)
        self.assertIn("## `overview`: whole-video summary", baseline_actor.messages[0]["content"])

    @patch("videoseek.agent.load_subtitles")
    @patch("videoseek.agent.VideoReader")
    @patch("videoseek.tools.answer.call_llm_api")
    @patch("videoseek.tools.inspect.call_llm_api")
    @patch("videoseek.agent.call_llm_api")
    def test_6_end_to_end_dual_agent_reflexion_workflow(
        self,
        mock_agent_llm,
        mock_inspect_llm,
        mock_answer_llm,
        mock_vr_cls,
        mock_load_subs,
    ):
        """
        Test 6: End-to-End run of Actor Agent + Reflexion Evaluator Agent:
        Step 1: Macro inspect(0, 120, fps=0.5) -> Evaluator marks INSUFFICIENT & recommends [44, 47] @ fps=8
        Step 2: Micro inspect(44, 47, fps=8)   -> Evaluator marks SUFFICIENT & commands calling answer
        Step 3: Actor calls answer()           -> Final answer 'BI MAT' returned with reflections saved in trajectory.
        """
        mock_vr_cls.return_value = FakeVideoReader(total_frames=3600, fps=30.0)
        mock_load_subs.return_value = []

        # Sequence of calls to videoseek.agent.call_llm_api:
        # 1. Step 1 Actor Thought
        # 2. Step 1 Action Parser -> inspect(0, 120, 0.5)
        # 3. Step 1 Evaluator Reflexion -> INSUFFICIENT, zoom to [44, 47] fps=8
        # 4. Step 2 Actor Thought
        # 5. Step 2 Action Parser -> inspect(44, 47, 8)
        # 6. Step 2 Evaluator Reflexion -> SUFFICIENT, call answer immediately
        # 7. Step 3 Actor Thought
        # 8. Step 3 Action Parser -> answer()
        mock_agent_llm.side_effect = [
            _make_llm_response(
                content="Step 1: Cold-start whole-video scan using inspect(0, 120, fps=0.5)."
            ),
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_inspect_1",
                        "function": {
                            "name": "inspect",
                            "arguments": json.dumps(
                                {
                                    "query": "Scan for red flashing text",
                                    "start_time": 0,
                                    "end_time": 120,
                                    "fps": 0.5,
                                }
                            ),
                        },
                    }
                ]
            ),
            _make_llm_response(
                content=(
                    "- **Sufficiency Status**: `INSUFFICIENT`\n"
                    "- **Evidence Audit**: Red box spotted around 45.0s, but exact letters need dense verification.\n"
                    "- **Reflexion Directive**: Call inspect with recommended range [44.0, 47.0] and fps=8."
                )
            ),
            _make_llm_response(
                content="Step 2: Following Evaluator directive, call inspect(44.0, 47.0, fps=8) to read the red text."
            ),
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_inspect_2",
                        "function": {
                            "name": "inspect",
                            "arguments": json.dumps(
                                {
                                    "query": "Read exact red text in box",
                                    "start_time": 44.0,
                                    "end_time": 47.0,
                                    "fps": 8,
                                }
                            ),
                        },
                    }
                ]
            ),
            _make_llm_response(
                content=(
                    "- **Sufficiency Status**: `SUFFICIENT`\n"
                    "- **Evidence Audit**: At 44.50s-46.50s, the red text is unambiguously confirmed as 'BI MAT'.\n"
                    "- **Reflexion Directive**: Call answer immediately."
                )
            ),
            _make_llm_response(
                content="Step 3: Evaluator confirmed SUFFICIENT. Calling answer tool."
            ),
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_answer_3",
                        "function": {"name": "answer", "arguments": "{}"},
                    }
                ]
            ),
        ]

        mock_inspect_llm.side_effect = [
            _make_llm_response(
                content=json.dumps(
                    {
                        "frames": [
                            {
                                "timestamp": "45.0s",
                                "description": "A red rectangle appears in the center.",
                            }
                        ]
                    }
                )
            ),
            _make_llm_response(
                content="At 44.50s - 46.50s, the uppercase red text inside the box reads 'BI MAT'."
            ),
        ]
        mock_answer_llm.return_value = _make_llm_response(content="BI MAT")

        agent = VideoSeekAgent(
            config=self.config,
            video_path="synthetic.mp4",
            subtitle_path="synthetic.srt",
            output_dir="./outputs",
            tools=["inspect"],
            verbose=False,
        )
        traj = agent.run("What is the flashing red text in the video?")

        self.assertEqual(traj.final_answer, "BI MAT")
        self.assertEqual(traj.finish_reason, "stop")
        self.assertEqual(len(traj.steps), 3)
        self.assertEqual(traj.steps[0].action.function_name, "inspect")
        self.assertIn("INSUFFICIENT", traj.steps[0].reflection)
        self.assertEqual(traj.steps[1].action.function_name, "inspect")
        self.assertEqual(traj.steps[1].action.parameters["fps"], 8)
        self.assertIn("SUFFICIENT", traj.steps[1].reflection)
        self.assertEqual(traj.steps[2].action.function_name, "answer")

        # Ensure trajectory JSON serialization is clean and includes reflections
        traj_dict = traj.to_dict()
        serialized = json.dumps(traj_dict, ensure_ascii=False)
        self.assertIn("reflection", serialized)
        for s in traj_dict["steps"]:
            self.assertNotIn("vr", s["action"]["parameters"])
            self.assertNotIn("subtitles", s["action"]["parameters"])

    @patch("videoseek.tools.inspect.call_llm_api")
    def test_7_inspect_boundary_and_invalid_parameters(self, mock_llm):
        """Test 7: Edge cases in execute_inspect (start_time >= end_time, negative start_time, end_time > duration, fps <= 0)."""
        mock_llm.return_value = _make_llm_response(
            content="Boundary inspection succeeded."
        )
        vr = FakeVideoReader(total_frames=900, fps=30.0)  # 30.0s video

        # Case A: start_time >= end_time and beyond duration (35.0 >= 32.0 > 30.0s), plus fps <= 0
        out = execute_inspect(
            self.config,
            {
                "query": "Check reversed and out-of-bounds window",
                "start_time": 35.0,
                "end_time": 32.0,
                "fps": -2.0,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(out, "Boundary inspection succeeded.")
        sent_content = mock_llm.call_args.kwargs["messages"][0]["content"]
        self.assertIn("29.0s - 30.0s", sent_content[0]["text"])
        image_parts = [p for p in sent_content if p["type"] == "image_url"]
        # fps <= 0 defaults to 1.0 -> 1.0s window * 1.0 fps = 1 frame
        self.assertEqual(len(image_parts), 1)

        # Case B: negative start_time (-5.0s) and zero-width window (-5.0s to -5.0s) at fps=4
        out_neg = execute_inspect(
            self.config,
            {
                "query": "Check negative start_time",
                "start_time": -5.0,
                "end_time": -5.0,
                "fps": 4.0,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(out_neg, "Boundary inspection succeeded.")
        sent_content_neg = mock_llm.call_args.kwargs["messages"][0]["content"]
        self.assertIn("0.0s - 1.0s", sent_content_neg[0]["text"])
        image_parts_neg = [p for p in sent_content_neg if p["type"] == "image_url"]
        self.assertEqual(len(image_parts_neg), 4)

    @patch("videoseek.agent.load_subtitles")
    @patch("videoseek.agent.VideoReader")
    @patch("videoseek.tools.answer.call_llm_api")
    @patch("videoseek.tools.focus.call_llm_api")
    @patch("videoseek.tools.overview.call_llm_api")
    @patch("videoseek.agent.call_llm_api")
    def test_8_pydantic_v2_tool_calls_and_baseline_3tools_run(
        self,
        mock_agent_llm,
        mock_overview_llm,
        mock_focus_llm,
        mock_answer_llm,
        mock_vr_cls,
        mock_load_subs,
    ):
        """
        Test 8: Verify Pydantic v2 tool_calls extraction (string .json() and object attributes)
        and end-to-end execution of Baseline 3-tools mode (`overview`, `skim`, `focus` with `reflexion_enabled=False`).
        """
        from videoseek.agent import _extract_tool_calls_from_message

        # 1. Verify Pydantic v2 string .json() and object attribute extraction
        class PydanticV2StrJsonMsg:
            tool_calls = None

            def json(self):
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call_v2_1",
                                "function": {
                                    "name": "overview",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                )

        class ObjFn:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class ObjToolCall:
            def __init__(self, cid, name, arguments):
                self.id = cid
                self.function = ObjFn(name, arguments)

        class ObjMsg:
            def __init__(self, tool_calls):
                self.tool_calls = tool_calls

        extracted_str = _extract_tool_calls_from_message(PydanticV2StrJsonMsg())
        self.assertEqual(len(extracted_str), 1)
        self.assertEqual(extracted_str[0]["function"]["name"], "overview")

        extracted_obj = _extract_tool_calls_from_message(
            ObjMsg([ObjToolCall("call_obj_1", "focus", '{"query":"q","start_time":10,"end_time":14}')])
        )
        self.assertEqual(len(extracted_obj), 1)
        self.assertEqual(extracted_obj[0].function.name, "focus")

        # 2. End-to-End Baseline 3-tools run with reflexion_enabled=False
        mock_vr_cls.return_value = FakeVideoReader(total_frames=3600, fps=30.0)
        mock_load_subs.return_value = []

        resp_parse_overview = MagicMock()
        resp_parse_overview.choices = [MagicMock(message=PydanticV2StrJsonMsg())]

        resp_parse_focus = MagicMock()
        resp_parse_focus.choices = [
            MagicMock(
                message=ObjMsg(
                    [
                        ObjToolCall(
                            "call_focus_base",
                            "focus",
                            json.dumps(
                                {
                                    "query": "Check red text",
                                    "start_time": 44.0,
                                    "end_time": 47.0,
                                }
                            ),
                        )
                    ]
                )
            )
        ]

        resp_parse_answer = MagicMock()
        resp_parse_answer.choices = [
            MagicMock(message=ObjMsg([ObjToolCall("call_ans_base", "answer", "{}")]))
        ]

        mock_agent_llm.side_effect = [
            _make_llm_response(content="Step 1: Call overview."),
            resp_parse_overview,
            _make_llm_response(content="Step 2: Call focus on 44s-47s."),
            resp_parse_focus,
            _make_llm_response(content="Step 3: Call answer."),
            resp_parse_answer,
        ]

        mock_overview_llm.return_value = _make_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "45.0s",
                            "description": "Something red appears around 45s.",
                        }
                    ]
                }
            )
        )
        mock_focus_llm.return_value = _make_llm_response(
            content="The text at 45s is BI MAT."
        )
        mock_answer_llm.return_value = _make_llm_response(content="BI MAT")

        baseline_cfg = {**self.config, "reflexion_enabled": False}
        baseline_agent = VideoSeekAgent(
            config=baseline_cfg,
            video_path="dummy.mp4",
            subtitle_path="dummy.srt",
            output_dir="./outputs/baseline",
            tools=["overview", "skim", "focus"],
            verbose=False,
        )
        traj = baseline_agent.run("What is the red text?")

        self.assertEqual(traj.final_answer, "BI MAT")
        self.assertEqual(traj.finish_reason, "stop")
        self.assertEqual(len(traj.steps), 3)
        self.assertEqual(
            [s.action.function_name for s in traj.steps],
            ["overview", "focus", "answer"],
        )
        # Verify no reflection is attached when reflexion_enabled=False
        for s in traj.steps:
            self.assertIsNone(s.reflection)


if __name__ == "__main__":
    unittest.main()
