import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from videoseek.core.query_analyzer import QueryAnalyzer, QueryProfile
from videoseek.tools.focus import execute_focus, focus_tool
from videoseek.tools.overview import execute_overview, _sample_hybrid_anomaly_frames
from videoseek.agent import VideoSeekAgent, _extract_tool_calls_from_message
from config import general_config, prompts_config


class DummyBatch:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


class FakeVideoReader:
    """Lightweight mock of decord.VideoReader with an injected visual anomaly."""

    def __init__(
        self,
        total_frames: int = 36000,
        fps: float = 30.0,
        anomaly_range: tuple = (12600, 12690),
    ):
        self.total_frames = total_frames
        self.fps = fps
        self.anomaly_range = anomaly_range

    def __len__(self) -> int:
        return self.total_frames

    def get_avg_fps(self) -> float:
        return self.fps

    def get_batch(self, indices) -> DummyBatch:
        frames = []
        for idx in indices:
            idx = int(idx)
            frame = np.full((64, 64, 3), 40, dtype=np.uint8)
            if self.anomaly_range[0] <= idx <= self.anomaly_range[1]:
                # High-contrast bright red anomaly block ("BI MAT")
                frame[16:48, 8:56, 0] = 255
                frame[16:48, 8:56, 1] = 0
                frame[16:48, 8:56, 2] = 0
            frames.append(frame)
        return DummyBatch(np.stack(frames, axis=0))


def _make_mock_llm_response(content: str = "", tool_calls: list = None):
    mock_resp = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_msg.tool_calls = tool_calls or []
    # Simulate Pydantic v2 where .json() returns a JSON string (not a dict)
    mock_msg.json.return_value = json.dumps({"content": content, "tool_calls": tool_calls or []})
    mock_resp.choices = [MagicMock(message=mock_msg)]
    return mock_resp


class TestModule2AdaptivePerception(unittest.TestCase):
    def setUp(self):
        self.config = {
            **general_config,
            **prompts_config,
            "model_name": "gpt-5",
            "api_base": "https://api.openai.com/v1",
            "api_key": "test-key",
            "api_version": None,
            "max_tokens": 2048,
            "max_steps": 5,
            "reasoning_effort": "low",
            "seed": 42,
            "temperature": 1.0,
            "adaptive_fps_enabled": True,
            "anomaly_overview_enabled": True,
            "max_focus_frames_cap": 24,
        }

    @patch("videoseek.core.query_analyzer.call_llm_api")
    def test_1_query_analyzer_4_types_and_safe_coercion(self, mock_llm):
        """Test 1: LLM JSON Router for all 4 query_types + safe type coercion for strings."""
        analyzer = QueryAnalyzer(self.config)

        scenarios = [
            {
                "question": "What is the main storyline of the video?",
                "llm_json": {
                    "query_type": "global_narrative",
                    "recommended_fps": "1",
                    "enable_anomaly_scan": "false",
                    "target_cues": ["storyline", "summary"],
                    "planning_hint": "Call overview first.",
                },
                "expected_type": "global_narrative",
                "expected_fps": 1.0,
                "expected_scan": False,
            },
            {
                "question": "What happens between 02:10 and 02:15?",
                "llm_json": {
                    "query_type": "explicit_temporal",
                    "recommended_fps": "4",
                    "enable_anomaly_scan": "0",
                    "target_cues": ["action at 02:10"],
                    "planning_hint": "Directly focus on 130s-135s at 4 FPS.",
                },
                "expected_type": "explicit_temporal",
                "expected_fps": 4.0,
                "expected_scan": False,
            },
            {
                "question": "Which hand does the chef use to pick up the small spoon?",
                "llm_json": {
                    "query_type": "implicit_fine_grained",
                    "recommended_fps": 4,
                    "enable_anomaly_scan": False,
                    "target_cues": ["chef hand", "small spoon"],
                    "planning_hint": "Localize kitchen scene then focus at 4 FPS.",
                },
                "expected_type": "implicit_fine_grained",
                "expected_fps": 4.0,
                "expected_scan": False,
            },
            {
                "question": "Chữ màu đỏ xuất hiện chớp nhoáng trong video là gì?",
                "llm_json": {
                    "query_type": "anomaly_ocr",
                    "recommended_fps": "8.0",
                    "enable_anomaly_scan": "true",
                    "target_cues": ["red text", "BI MAT"],
                    "planning_hint": "Use overview with anomaly scan then focus at 8 FPS.",
                },
                "expected_type": "anomaly_ocr",
                "expected_fps": 8.0,
                "expected_scan": True,
            },
        ]

        for case in scenarios:
            with self.subTest(query_type=case["expected_type"]):
                mock_llm.return_value = _make_mock_llm_response(
                    content=json.dumps(case["llm_json"])
                )
                profile = analyzer.analyze(case["question"])
                self.assertEqual(profile.query_type, case["expected_type"])
                self.assertEqual(profile.recommended_fps, case["expected_fps"])
                self.assertEqual(profile.enable_anomaly_scan, case["expected_scan"])
                self.assertEqual(
                    mock_llm.call_args.kwargs.get("reasoning_effort"), "low"
                )
                self.assertTrue(mock_llm.call_args.kwargs.get("return_json"))

    @patch("videoseek.core.query_analyzer.call_llm_api")
    def test_2_query_analyzer_fallback_on_errors(self, mock_llm):
        """Test 2: LLM JSON Router safe fallback on None, malformed JSON, or missing fields."""
        analyzer = QueryAnalyzer(self.config)

        # Subcase A: API returns None
        mock_llm.return_value = None
        p_none = analyzer.analyze("What is the video about?")
        self.assertEqual(p_none.query_type, "global_narrative")
        self.assertEqual(p_none.recommended_fps, 1)
        self.assertFalse(p_none.enable_anomaly_scan)

        # Subcase B: Malformed JSON syntax
        mock_llm.return_value = _make_mock_llm_response(content="{broken_json: true")
        p_bad_json = analyzer.analyze("What is the video about?")
        self.assertEqual(p_bad_json.query_type, "global_narrative")
        self.assertEqual(p_bad_json.recommended_fps, 1)
        self.assertFalse(p_bad_json.enable_anomaly_scan)

        # Subcase C: Missing fields / invalid values in JSON object
        mock_llm.return_value = _make_mock_llm_response(
            content=json.dumps({"query_type": "unknown_type", "recommended_fps": -5})
        )
        p_missing = analyzer.analyze("Some question")
        self.assertEqual(p_missing.query_type, "global_narrative")
        self.assertEqual(p_missing.recommended_fps, 1)
        self.assertFalse(p_missing.enable_anomaly_scan)
        self.assertEqual(p_missing.target_cues, [])

        # Subcase D: Empty question string
        p_empty = analyzer.analyze("   ")
        self.assertEqual(p_empty.query_type, "global_narrative")

    @patch("videoseek.tools.focus.call_llm_api")
    def test_3_focus_schema_and_timestamp_formatting(self, mock_llm):
        """Test 3: Focus tool strict schema and .2f vs .1f timestamp formatting."""
        # Check schema
        self.assertTrue(focus_tool["function"]["strict"])
        params = focus_tool["function"]["parameters"]
        self.assertIn("fps", params["properties"])
        self.assertEqual(
            params["required"], ["query", "start_time", "end_time", "fps"]
        )
        self.assertFalse(params["additionalProperties"])

        mock_llm.return_value = _make_mock_llm_response(content="BI MAT")
        vr = FakeVideoReader(total_frames=36000, fps=30.0)

        # Case A: fps=8 on [420.0, 423.0] -> 24 frames, .2f timestamp formatting
        res_8 = execute_focus(
            self.config,
            {
                "query": "Read the flashing red text",
                "start_time": 420.0,
                "end_time": 423.0,
                "fps": 8,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(res_8, "BI MAT")
        sent_content_8 = mock_llm.call_args.kwargs["messages"][0]["content"]
        image_parts_8 = [p for p in sent_content_8 if p["type"] == "image_url"]
        ts_texts_8 = [p["text"] for p in sent_content_8 if p["type"] == "text"][1:-1]
        self.assertEqual(len(image_parts_8), 24)
        self.assertEqual(len(ts_texts_8), 24)
        for ts in ts_texts_8:
            self.assertTrue(ts.endswith("s"))
            decimals = ts[:-1].split(".")[1]
            self.assertEqual(len(decimals), 2, f"Expected .2f format, got {ts}")

        # Case B: fps=1 on [420.0, 423.0] -> 3 frames, .1f timestamp formatting
        res_1 = execute_focus(
            self.config,
            {
                "query": "General scene check",
                "start_time": 420.0,
                "end_time": 423.0,
                "fps": 1,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(res_1, "BI MAT")
        sent_content_1 = mock_llm.call_args.kwargs["messages"][0]["content"]
        image_parts_1 = [p for p in sent_content_1 if p["type"] == "image_url"]
        ts_texts_1 = [p["text"] for p in sent_content_1 if p["type"] == "text"][1:-1]
        self.assertEqual(len(image_parts_1), 3)
        for ts in ts_texts_1:
            self.assertTrue(ts.endswith("s"))
            decimals = ts[:-1].split(".")[1]
            self.assertEqual(len(decimals), 1, f"Expected .1f format, got {ts}")

    @patch("videoseek.tools.focus.call_llm_api")
    def test_4_focus_edge_cases(self, mock_llm):
        """Test 4: Focus edge cases (start_time >= end_time, > video_duration, max_focus_frames_cap=24)."""
        mock_llm.return_value = _make_mock_llm_response(content="Edge case OK")
        vr = FakeVideoReader(total_frames=900, fps=30.0)  # 30.0s video

        # Case A: start_time >= end_time and exceeding video duration (35.0 >= 32.0 > 30.0s)
        out1 = execute_focus(
            self.config,
            {
                "query": "Check boundary",
                "start_time": 35.0,
                "end_time": 32.0,
                "fps": 8,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(out1, "Edge case OK")
        content1 = mock_llm.call_args.kwargs["messages"][0]["content"]
        images1 = [p for p in content1 if p["type"] == "image_url"]
        self.assertEqual(len(images1), 8)  # 1.0s clamped window * 8 FPS = 8 frames

        # Case B: Requesting a long 10s window at fps=8 -> 80 frames requested, capped at max_focus_frames_cap=24
        out2 = execute_focus(
            self.config,
            {
                "query": "Check max cap",
                "start_time": 5.0,
                "end_time": 15.0,
                "fps": 8,
                "vr": vr,
                "subtitles": [],
            },
        )
        self.assertEqual(out2, "Edge case OK")
        content2 = mock_llm.call_args.kwargs["messages"][0]["content"]
        images2 = [p for p in content2 if p["type"] == "image_url"]
        self.assertEqual(len(images2), 24)

    def test_5_hybrid_anomaly_overview_20min_video_at_420_423s(self):
        """
        Test 5: 20-minute video (1200s @ 30 FPS = 36,000 frames) with a 3-second red text anomaly
        at 420s-423s (frames 12600-12690).
        Verify:
        1) Standard uniform sampling (64 frames) completely misses [12600, 12690].
        2) Hybrid anomaly sampling (_sample_hybrid_anomaly_frames) successfully captures frames in [12600, 12690].
        """
        total_frames = 36000
        fps = 30.0
        anomaly_start, anomaly_end = 12600, 12690  # 420.0s to 423.0s
        num_frames = 64

        # 1. Verify standard uniform linspace misses [12600, 12690]
        uniform_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        uniform_hits = np.any(
            (uniform_indices >= anomaly_start) & (uniform_indices <= anomaly_end)
        )
        self.assertFalse(
            uniform_hits,
            f"Uniform sampling should miss [12600, 12690], but got {uniform_indices[(uniform_indices >= 12300) & (uniform_indices <= 13000)]}",
        )

        # 2. Verify Hybrid Anomaly Overview catches [12600, 12690]
        vr = FakeVideoReader(
            total_frames=total_frames,
            fps=fps,
            anomaly_range=(anomaly_start, anomaly_end),
        )
        hybrid_indices = _sample_hybrid_anomaly_frames(
            vr, total_frames=total_frames, num_frames=num_frames
        )
        self.assertEqual(len(hybrid_indices), num_frames)
        self.assertEqual(len(hybrid_indices) % 8, 0)
        self.assertTrue(np.all(np.diff(hybrid_indices) >= 0))

        hybrid_hits = np.any(
            (hybrid_indices >= anomaly_start) & (hybrid_indices <= anomaly_end)
        )
        self.assertTrue(
            hybrid_hits,
            "Hybrid anomaly sampling MUST capture at least one frame inside [12600, 12690] (420s-423s)!",
        )

    @patch("videoseek.tools.overview.call_llm_api")
    def test_6_query_conditioned_overview_prompt(self, mock_llm):
        """Test 6: Verify execute_overview injects question and target_cues into the VLM prompt."""
        mock_llm.return_value = _make_mock_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "421.2s",
                            "description": "Red flashing text BI MAT appears on screen.",
                        }
                    ]
                }
            )
        )
        vr = FakeVideoReader(
            total_frames=36000, fps=30.0, anomaly_range=(12600, 12690)
        )
        profile = QueryProfile(
            query_type="anomaly_ocr",
            recommended_fps=8,
            enable_anomaly_scan=True,
            target_cues=["chữ màu đỏ", "BI MAT"],
            planning_hint="Scan anomaly then focus at 8 FPS.",
        )
        question = "Dòng chữ màu đỏ xuất hiện chớp nhoáng ở phút thứ 7 là gì?"
        out = execute_overview(
            self.config,
            {
                "vr": vr,
                "subtitles": [],
                "question": question,
                "query_profile": profile,
            },
        )
        self.assertIn("421.2s: Red flashing text BI MAT", out)

        sent_content = mock_llm.call_args.kwargs["messages"][0]["content"]
        first_text = sent_content[0]["text"]
        last_text = sent_content[-1]["text"]
        self.assertIn("hybrid uniform + visual anomaly peak detection", first_text)
        self.assertIn(question, last_text)
        self.assertIn("chữ màu đỏ, BI MAT", last_text)

    @patch("videoseek.agent.load_subtitles")
    @patch("videoseek.agent.VideoReader")
    @patch("videoseek.tools.answer.call_llm_api")
    @patch("videoseek.tools.focus.call_llm_api")
    @patch("videoseek.tools.overview.call_llm_api")
    @patch("videoseek.agent.call_llm_api")
    @patch("videoseek.core.query_analyzer.call_llm_api")
    def test_7_end_to_end_agent_and_clean_trajectory(
        self,
        mock_router_llm,
        mock_agent_llm,
        mock_overview_llm,
        mock_focus_llm,
        mock_answer_llm,
        mock_vr_cls,
        mock_load_subs,
    ):
        """
        Test 7: End-to-End VideoSeekAgent.run() through overview -> focus(fps=8) -> answer,
        verifying trajectory.to_dict() is 100% JSON serializable and does not leak internal keys.
        """
        mock_vr_cls.return_value = FakeVideoReader(
            total_frames=36000, fps=30.0, anomaly_range=(12600, 12690)
        )
        mock_load_subs.return_value = []

        # 1. Router response
        mock_router_llm.return_value = _make_mock_llm_response(
            content=json.dumps(
                {
                    "query_type": "anomaly_ocr",
                    "recommended_fps": 8,
                    "enable_anomaly_scan": True,
                    "target_cues": ["red text", "BI MAT"],
                    "planning_hint": "Call overview then focus with fps=8.",
                }
            )
        )

        # 2. Agent loop LLM calls:
        # Step 1 Thought -> Step 1 Parse Action (overview) ->
        # Step 2 Thought -> Step 2 Parse Action (focus, fps=8) ->
        # Step 3 Thought -> Step 3 Parse Action (answer)
        mock_agent_llm.side_effect = [
            _make_mock_llm_response(
                content="Step 1: Call overview to scan for the red text anomaly."
            ),
            _make_mock_llm_response(
                tool_calls=[
                    {
                        "id": "call_overview_1",
                        "function": {"name": "overview", "arguments": "{}"},
                    }
                ]
            ),
            _make_mock_llm_response(
                content="Step 2: Overview detected red text around 421.2s. Let's focus on 420.0s-423.0s with fps=8."
            ),
            _make_mock_llm_response(
                tool_calls=[
                    {
                        "id": "call_focus_2",
                        "function": {
                            "name": "focus",
                            "arguments": json.dumps(
                                {
                                    "query": "Read the red flashing text",
                                    "start_time": 420.0,
                                    "end_time": 423.0,
                                    "fps": 8,
                                }
                            ),
                        },
                    }
                ]
            ),
            _make_mock_llm_response(
                content="Step 3: The red text is clearly 'BI MAT'. Ready to answer."
            ),
            _make_mock_llm_response(
                tool_calls=[
                    {
                        "id": "call_answer_3",
                        "function": {"name": "answer", "arguments": "{}"},
                    }
                ]
            ),
        ]

        # 3. Tool responses
        mock_overview_llm.return_value = _make_mock_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "421.2s",
                            "description": "Bright red text appears briefly on screen.",
                        }
                    ]
                }
            )
        )
        mock_focus_llm.return_value = _make_mock_llm_response(
            content="The flashing red text at 420.0s-423.0s reads 'BI MAT'."
        )
        mock_answer_llm.return_value = _make_mock_llm_response(content="BI MAT")

        agent = VideoSeekAgent(
            config=self.config,
            video_path="dummy_20min.mp4",
            subtitle_path="dummy.srt",
            output_dir="outputs",
            tools=["overview", "skim", "focus"],
            verbose=False,
        )

        trajectory = agent.run("Chữ màu đỏ xuất hiện chớp nhoáng từ 420s đến 423s là gì?")

        self.assertEqual(trajectory.final_answer, "BI MAT")
        self.assertEqual(trajectory.finish_reason, "stop")
        self.assertEqual(len(trajectory.steps), 3)
        self.assertEqual(trajectory.steps[0].action.function_name, "overview")
        self.assertEqual(trajectory.steps[1].action.function_name, "focus")
        self.assertEqual(trajectory.steps[1].action.parameters["fps"], 8)
        self.assertEqual(trajectory.steps[2].action.function_name, "answer")

        # Verify clean trajectory serialization without any leaked internal objects
        traj_dict = trajectory.to_dict()
        serialized = json.dumps(traj_dict, ensure_ascii=False)
        self.assertIn("BI MAT", serialized)

        for step_dict in traj_dict["steps"]:
            params = step_dict["action"]["parameters"]
            self.assertNotIn("vr", params)
            self.assertNotIn("subtitles", params)
            self.assertNotIn("question", params)
            self.assertNotIn("query_profile", params)

        # Verify OpenAI chat-completions tool_calls arguments in agent.messages are valid JSON strings
        assistant_tool_msgs = [
            m for m in agent.messages if m.get("role") == "assistant" and "tool_calls" in m
        ]
        self.assertEqual(len(assistant_tool_msgs), 2)
        for m in assistant_tool_msgs:
            for tc in m["tool_calls"]:
                parsed_args = json.loads(tc["function"]["arguments"])
                self.assertIsInstance(parsed_args, dict)

    @patch("videoseek.agent.load_subtitles")
    @patch("videoseek.agent.VideoReader")
    @patch("videoseek.tools.answer.call_llm_api")
    @patch("videoseek.tools.overview.call_llm_api")
    @patch("videoseek.agent.call_llm_api")
    @patch("videoseek.core.query_analyzer.call_llm_api")
    def test_8_pydantic_v2_tool_calls_and_step0_fallback(
        self,
        mock_router_llm,
        mock_agent_llm,
        mock_overview_llm,
        mock_answer_llm,
        mock_vr_cls,
        mock_load_subs,
    ):
        """
        Test 8: Verify:
        1) _extract_tool_calls_from_message & __parse_actions work when message.json() returns a JSON string (Pydantic v2)
           and when tool_calls are objects with .function.name and .function.arguments attributes.
        2) Step 0 fallback: when THOUGHT says "Không đủ bằng chứng để xác định dòng chữ màu đỏ." and parser returns [],
           the agent automatically falls back to calling `overview`, records the Observation, and formats assistant tool_calls
           arguments as valid JSON strings (json.loads succeeds).
        """
        # Part A: Test _extract_tool_calls_from_message with Pydantic v2 JSON string
        class PydanticV2MsgWithJsonStr:
            tool_calls = None

            def json(self):
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call_pydantic_v2",
                                "function": {
                                    "name": "focus",
                                    "arguments": json.dumps(
                                        {
                                            "query": "Read red text",
                                            "start_time": 420.0,
                                            "end_time": 423.0,
                                            "fps": 8,
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                )

        extracted_v2 = _extract_tool_calls_from_message(PydanticV2MsgWithJsonStr())
        self.assertEqual(len(extracted_v2), 1)
        self.assertEqual(extracted_v2[0]["function"]["name"], "focus")

        # Part B: Test object-based tool_calls with .function.name and .function.arguments
        class DummyFnObj:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class DummyToolCallObj:
            def __init__(self, call_id, name, arguments):
                self.id = call_id
                self.function = DummyFnObj(name, arguments)

        class MsgWithObjToolCalls:
            def __init__(self):
                self.tool_calls = [
                    DummyToolCallObj(
                        "call_obj_1",
                        "focus",
                        '{"query": "Check red text", "start_time": 44.0, "end_time": 47.0, "fps": 8}',
                    )
                ]

        extracted_obj = _extract_tool_calls_from_message(MsgWithObjToolCalls())
        self.assertEqual(len(extracted_obj), 1)
        self.assertEqual(extracted_obj[0].function.name, "focus")

        # Part C: Verify __parse_actions handles both Pydantic v2 json() str and object tool_calls,
        # and Step 0 fallback during agent.run() when THOUGHT is "Không đủ bằng chứng để xác định dòng chữ màu đỏ."
        mock_vr_cls.return_value = FakeVideoReader(
            total_frames=3600, fps=30.0, anomaly_range=(1320, 1410)
        )
        mock_load_subs.return_value = []

        mock_router_llm.return_value = _make_mock_llm_response(
            content=json.dumps(
                {
                    "query_type": "anomaly_ocr",
                    "recommended_fps": 8,
                    "enable_anomaly_scan": True,
                    "target_cues": ["dòng chữ màu đỏ"],
                    "planning_hint": "Call overview first.",
                }
            )
        )

        # Create a response whose message.json() returns a str with empty tool_calls (triggering step 0 fallback)
        step0_empty_parser_resp = MagicMock()
        step0_msg = MagicMock()
        step0_msg.tool_calls = None
        step0_msg.model_dump.return_value = {"tool_calls": []}
        step0_msg.json.return_value = json.dumps({"tool_calls": []})
        step0_empty_parser_resp.choices = [MagicMock(message=step0_msg)]

        # Create a response for step 1 parser using object-style tool_calls for `answer`
        step1_obj_parser_resp = MagicMock()
        step1_msg = MsgWithObjToolCalls()
        step1_msg.tool_calls = [DummyToolCallObj("call_ans_obj", "answer", "{}")]
        step1_obj_parser_resp.choices = [MagicMock(message=step1_msg)]

        mock_agent_llm.side_effect = [
            # Step 0 THOUGHT: states insufficient evidence without naming a tool
            _make_mock_llm_response(
                content="Không đủ bằng chứng để xác định dòng chữ màu đỏ."
            ),
            # Step 0 ACTION PARSER: returns empty tool_calls (Pydantic v2 json str) -> triggers step 0 overview fallback
            step0_empty_parser_resp,
            # Step 1 THOUGHT: ready to answer after overview observation
            _make_mock_llm_response(
                content="Overview showed the red text is 'BI MAT'. Calling answer."
            ),
            # Step 1 ACTION PARSER: object-style tool_call for `answer`
            step1_obj_parser_resp,
        ]

        mock_overview_llm.return_value = _make_mock_llm_response(
            content=json.dumps(
                {
                    "frames": [
                        {
                            "timestamp": "45.0s",
                            "description": "Red flashing text 'BI MAT' is visible.",
                        }
                    ]
                }
            )
        )
        mock_answer_llm.return_value = _make_mock_llm_response(content="BI MAT")

        agent = VideoSeekAgent(
            config=self.config,
            video_path="dummy.mp4",
            subtitle_path="dummy.srt",
            output_dir="outputs",
            tools=["overview", "skim", "focus"],
            verbose=False,
        )

        # Also directly test __parse_actions with PydanticV2MsgWithJsonStr
        pydantic_v2_resp = MagicMock()
        pydantic_v2_resp.choices = [MagicMock(message=PydanticV2MsgWithJsonStr())]
        with patch("videoseek.agent.call_llm_api", return_value=pydantic_v2_resp):
            parsed_actions = agent._VideoSeekAgent__parse_actions(
                "Focus on 420s-423s at 8 FPS", step=1
            )
            self.assertEqual(len(parsed_actions), 1)
            self.assertEqual(parsed_actions[0].function_name, "focus")
            self.assertEqual(parsed_actions[0].parameters["fps"], 8)

        # Run full agent flow testing step 0 fallback
        traj = agent.run("Dòng chữ màu đỏ xuất hiện chớp nhoáng là gì?")
        self.assertEqual(traj.final_answer, "BI MAT")
        self.assertEqual(len(traj.steps), 2)
        self.assertEqual(traj.steps[0].action.function_name, "overview")
        self.assertIn("BI MAT", traj.steps[0].observation.outcome)
        self.assertEqual(traj.steps[1].action.function_name, "answer")

        # Verify assistant messages with tool_calls have valid JSON string in function.arguments
        assistant_tc_msgs = [
            m for m in agent.messages if m.get("role") == "assistant" and "tool_calls" in m
        ]
        self.assertGreaterEqual(len(assistant_tc_msgs), 1)
        for m in assistant_tc_msgs:
            for tc in m["tool_calls"]:
                args_str = tc["function"]["arguments"]
                self.assertIsInstance(args_str, str)
                parsed_args = json.loads(args_str)
                self.assertIsInstance(parsed_args, dict)


if __name__ == "__main__":
    unittest.main()
