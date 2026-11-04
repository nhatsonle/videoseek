import json
from typing import Any, List
from abc import ABC, abstractmethod

from decord import VideoReader

from .core import (
    Action,
    Observation,
    Trajectory,
    TrajectoryStep,
    QueryAnalyzer,
)
from .tools import DEFAULT_TOOL_REGISTRY
from .utils import call_llm_api, convert_to_free_form_text_representation, load_subtitles


def _extract_tool_calls_from_message(msg: Any) -> list:
    """
    Safely extract tool_calls from a litellm/OpenAI message object across
    Pydantic v1, Pydantic v2, dict, or JSON string representations.
    """
    if msg is None:
        return []

    # 1. Direct attribute access on message object
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return list(tool_calls)

    # 2. Dict access if msg is already a dict
    if isinstance(msg, dict):
        tc = msg.get("tool_calls")
        return list(tc) if tc else []

    # 3. Pydantic v2 model_dump()
    if hasattr(msg, "model_dump") and callable(msg.model_dump):
        try:
            dumped = msg.model_dump()
            if isinstance(dumped, dict) and dumped.get("tool_calls"):
                return list(dumped["tool_calls"])
        except Exception:
            pass

    # 4. msg.json() (returns str in Pydantic v2, or dict in some custom wrappers)
    if hasattr(msg, "json") and callable(msg.json):
        try:
            raw_json = msg.json()
            if isinstance(raw_json, str) and raw_json.strip():
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and parsed.get("tool_calls"):
                    return list(parsed["tool_calls"])
            elif isinstance(raw_json, dict) and raw_json.get("tool_calls"):
                return list(raw_json["tool_calls"])
        except Exception:
            pass

    return []


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: List[dict] = []
        self.final_answer = None
        self.question = None

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None

    @abstractmethod
    def construct_initial_messages(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSeekAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        tools: list,
        verbose: bool = False,
    ):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = self.tool_registry.resolve_tools(tools + ["answer"])
        self.output_dir = output_dir
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

        # Query-Adaptive Perception Analyzer
        self.query_analyzer = QueryAnalyzer(config=self.config)
        self.query_profile = None

        # LLM config
        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_steps = config["max_steps"]
        self.max_tokens = config["max_tokens"]
        self.reasoning_effort = config["reasoning_effort"]
        self.seed = config["seed"]
        self.temperature = config["temperature"]

        # initial messages
        self.messages = self.construct_initial_messages()
        # trajectory
        self.trajectory_steps: List[TrajectoryStep] = []

    def reset(self):
        """
        Reset the agent.
        """
        super().reset()
        self.trajectory_steps = []
        self.query_profile = None

    def construct_initial_messages(self) -> List[dict]:
        """
        Build the initial [system, user] messages.
        Subclasses should return a list of chat messages that may contain placeholders.
        """
        system_prompt = self.config["SYSTEM_PROMPT"].format(
            overview_num_frames=self.config["frame_sampling_factor"] * self.config["overview_base"],
            skim_num_frames=self.config["frame_sampling_factor"] * self.config["skim_base"],
            focus_num_frames=self.config["frame_sampling_factor"] * self.config["focus_base"],
        )
        return [{"role": "system", "content": system_prompt}]

    def __parse_actions(self, thought: str, step: int = 0) -> List[Action]:
        """
        Parse the actions from the thought.
        Compatible with both Pydantic v1 and Pydantic v2 litellm responses.
        """
        actions = []

        try:
            parser_prompt = (
                f"Please call the appropriate tool(s) based on the following thought:\n{thought}\n\n"
                "Note: If the thought indicates that current information or subtitles are insufficient and no visual "
                "inspection has been performed yet, call the `overview` tool to inspect the video."
            )
            response = call_llm_api(
                messages=[
                    {
                        "role": "user",
                        "content": parser_prompt,
                    }
                ],
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                tool_choice="required",
                tools=self.tools,
                temperature=self.temperature,
            )

            if response is None or not getattr(response, "choices", None):
                if step == 0 and self.tool_registry.has_tool("overview"):
                    return [
                        Action(
                            function_name="overview",
                            parameters={},
                            function_id="call_overview_step0",
                        )
                    ]
                return []

            msg = response.choices[0].message
            tool_calls = _extract_tool_calls_from_message(msg)
            answer_call_id = None

            for tool_idx, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    fn_obj = tool_call.get("function", {}) or {}
                    function_id = tool_call.get("id", None)
                else:
                    fn_obj = getattr(tool_call, "function", None)
                    function_id = getattr(tool_call, "id", None)

                if isinstance(fn_obj, dict):
                    function_name = fn_obj.get("name", None)
                    raw_args = fn_obj.get("arguments", "{}")
                else:
                    function_name = getattr(fn_obj, "name", None) if fn_obj else None
                    raw_args = getattr(fn_obj, "arguments", "{}") if fn_obj else "{}"

                if isinstance(raw_args, dict):
                    parameters = dict(raw_args)
                elif isinstance(raw_args, str):
                    parameters = json.loads(raw_args) if raw_args.strip() else {}
                else:
                    parameters = {}

                if self.tool_registry.has_tool(function_name):
                    if function_name == "answer":
                        answer_call_id = len(actions)
                    actions.append(
                        Action(
                            function_name=function_name,
                            parameters=parameters,
                            function_id=function_id or f"call_{function_name}_{step}_{tool_idx}",
                        )
                    )
        except Exception as e:
            print(f"Error parsing actions: {e}")
            if step == 0 and self.tool_registry.has_tool("overview"):
                return [
                    Action(
                        function_name="overview",
                        parameters={},
                        function_id="call_overview_step0",
                    )
                ]
            return []

        non_answer_actions = [a for a in actions if a.function_name != "answer"]
        if non_answer_actions:
            actions = non_answer_actions
        elif len(actions) > 1:
            actions = [actions[0]]

        if len(actions) == 0 and step == 0 and self.tool_registry.has_tool("overview"):
            actions.append(
                Action(
                    function_name="overview",
                    parameters={},
                    function_id="call_overview_step0",
                )
            )

        return actions

    def __exec_action(self, action: Action) -> str:
        """
        Execute an action.
        """
        function_name = getattr(action, "function_name", None) if action else None
        parameters = getattr(action, "parameters", {}) if action else {}

        if function_name == "answer":
            exec_params = {"question": self.question, "messages": self.messages}
        else:
            parameters.update(
                {
                    "vr": self.vr,
                    "subtitles": self.subtitles,
                    "question": self.question,
                    "query_profile": self.query_profile,
                }
            )
            exec_params = parameters

        try:
            if self.tool_registry.has_tool(function_name):
                outcome = self.tool_registry.get_function(function_name)(
                    config=self.config, parameters=exec_params
                )
                if outcome is None:
                    outcome = "Tool execution failed."
                return outcome
            else:
                raise ValueError(f"Invalid function name: {function_name}")
        finally:
            if action and action.parameters is not None:
                action.parameters.pop("vr", None)
                action.parameters.pop("subtitles", None)
                action.parameters.pop("question", None)
                action.parameters.pop("query_profile", None)

    def run(self, question: str):
        self.reset()
        self.question = question
        subtitles_str = convert_to_free_form_text_representation(
            self.subtitles, content_type="subtitle"
        )

        # Query-Adaptive Analysis via LLM JSON Router
        query_hint_block = ""
        if self.config.get("adaptive_fps_enabled", True):
            self.query_profile = self.query_analyzer.analyze(question)
            if self.query_profile is not None:
                cues_str = (
                    ", ".join(self.query_profile.target_cues)
                    if self.query_profile.target_cues
                    else "None"
                )
                query_hint_block = (
                    "\n\nQuery Analysis Profile:\n"
                    f"- Query Type: {self.query_profile.query_type}\n"
                    f"- Recommended Focus FPS: {self.query_profile.recommended_fps:g}\n"
                    f"- Anomaly Scan Enabled: {self.query_profile.enable_anomaly_scan}\n"
                    f"- Target Cues: {cues_str}\n"
                    f"- Planning Hint: {self.query_profile.planning_hint}"
                )

        ############################
        # Input
        ############################
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles_str}\n\n"
                    f"Question:\n{question}"
                    f"{query_hint_block}"
                ),
            }
        )

        video_id = self.video_path.split("/")[-1].split(".")[0]
        if self.verbose:
            print("--------------------------------")
            print(f"Video ID: {video_id} ({self.duration:.01f}s)")
            print("--------------------------------")
            print(f"Question:")
            print(question)
            if self.query_profile is not None:
                print("--------------------------------")
                print(f"Query Profile: {self.query_profile.to_dict()}")
            print("--------------------------------")

        for step in range(self.max_steps):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Step [{step + 1} / {self.max_steps}]: "
                        "Please follow the Thinking Policy to do **reasoning over the current state** "
                        "and **explicitly write in plain text which tool (`overview`, `skim`, `focus`, or `answer`) and its parameters you plan to call next** "
                        "following the Tool Calling Policy and the Final Answer Policy. "
                        "(Note: Write your planned tool call in text during this Thought step; the system's action parser will execute the tool for you right after this step.) "
                        "No observation is needed to be provided in the response."
                    ),
                }
            )

            ############################################################
            # THOUGHT
            ############################################################
            response = call_llm_api(
                messages=self.messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
            )
            raw_thought = (
                response.choices[0].message.content
                if (response is not None and getattr(response, "choices", None))
                else None
            )
            thought = (
                raw_thought
                if (raw_thought and str(raw_thought).strip())
                else "I need to inspect the video using `overview` to gather initial visual evidence."
            )
            self.messages.append({"role": "assistant", "content": thought})
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] THOUGHT")
                print(thought)
                print("--------------------------------")

            ############################################################
            # ACTIONS
            ############################################################
            actions = self.__parse_actions(thought, step=step)
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] ACTIONS")
                print([str(action) for action in actions])
                print("--------------------------------")
            if len(actions) != 0 and actions[0].function_name != "answer":
                self.messages[-1]["tool_calls"] = [
                    {
                        "id": action.function_id,
                        "type": "function",
                        "function": {
                            "name": action.function_name,
                            "arguments": json.dumps(
                                action.parameters if action.parameters is not None else {},
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for action in actions
                ]

            ############################################################
            # OBSERVATIONS
            ############################################################
            for action in actions:
                try:
                    outcome = self.__exec_action(action)
                except Exception as e:
                    outcome = "Tool execution failed."
                if self.verbose:
                    print(f"[STEP {step+1} / {self.max_steps}] OBSERVATION")
                    print(outcome)
                    print("--------------------------------")
                observation = Observation(action=action, outcome=outcome)
                if action.parameters is not None:
                    action.parameters.pop("vr", None)
                    action.parameters.pop("subtitles", None)
                    action.parameters.pop("question", None)
                    action.parameters.pop("query_profile", None)
                self.trajectory_steps.append(
                    TrajectoryStep(
                        step_id=step + 1,
                        thought=thought,
                        action=action,
                        observation=observation,
                    )
                )
                if action.function_name == "answer":
                    self.final_answer = observation.outcome
                    break
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id,
                        "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                    }
                )
            
            if len(actions) == 0:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous thought did not specify which tool to run. "
                            "Please write a thought that explicitly states one tool name from [`overview`, `skim`, `focus`, `answer`] "
                            "and its parameters (e.g., calling `overview` to scan the video, or `focus` with `start_time`, `end_time`, `fps`, `query`) "
                            "so the system can execute it."
                        ),
                    }
                )
                continue

            ############################################################
            # STOP IF FINAL ANSWER IS FOUND
            ############################################################
            if self.final_answer is not None:
                break

        ############################################################
        # REACH MAX STEPS BUT NO FINAL ANSWER IS FOUND
        ############################################################
        if self.final_answer is None:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of steps. "
                        f"Question:\n{question}\n\n"
                        "If the question is a multiple-choice question, please directly answer with the option's letter from the given choices without any additional text."
                    ),
                }
            )
            response = call_llm_api(
                messages=self.messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
            )
            self.final_answer = (
                response.choices[0].message.content
                if (response is not None and getattr(response, "choices", None))
                else "Unable to determine final answer."
            )
            self.messages.append({"role": "assistant", "content": self.final_answer})
            return Trajectory(
                question=question,
                steps=self.trajectory_steps,
                final_answer=self.final_answer,
                finish_reason="reach_max_steps",
            )

        return Trajectory(
            question=question,
            steps=self.trajectory_steps,
            final_answer=self.final_answer,
            finish_reason="stop",
        )
