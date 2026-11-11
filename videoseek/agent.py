import json
from typing import Any, List, Optional
from abc import ABC, abstractmethod

from decord import VideoReader

from .core import Action, Observation, Trajectory, TrajectoryStep
from .tools import DEFAULT_TOOL_REGISTRY
from .utils import call_llm_api, convert_to_free_form_text_representation, load_subtitles


def _extract_tool_calls_from_message(msg: Any) -> list:
    """
    Safely extract tool_calls from a litellm/OpenAI message object across
    Pydantic v1, Pydantic v2, dict, or JSON string representations.
    """
    if msg is None:
        return []

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return list(tool_calls)

    if isinstance(msg, dict):
        tc = msg.get("tool_calls")
        return list(tc) if tc else []

    if hasattr(msg, "model_dump") and callable(msg.model_dump):
        try:
            dumped = msg.model_dump()
            if isinstance(dumped, dict) and dumped.get("tool_calls"):
                return list(dumped["tool_calls"])
        except Exception:
            pass

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


class ReflexionEvaluatorAgent:
    """
    Independent Reflexion Evaluator Agent (Agent 2) with its own System Prompt
    (`EVALUATOR_SYSTEM_PROMPT`) and isolated message context.
    Audits the Actor's exploration trajectory, verifies evidence sufficiency,
    detects reasoning loops, and provides parameter adjustment directives.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_tokens = config.get("max_tokens", 4096)
        self.reasoning_effort = config.get("reasoning_effort", "medium")
        self.seed = config.get("seed", 42)
        self.temperature = config.get("temperature", 1.0)

        overview_num_frames = config.get("frame_sampling_factor", 4) * config.get(
            "overview_base", 16
        )
        skim_num_frames = config.get("frame_sampling_factor", 4) * config.get(
            "skim_base", 4
        )
        focus_num_frames = config.get("frame_sampling_factor", 4) * config.get(
            "focus_base", 4
        )
        max_inspect_frames = config.get("max_inspect_frames", overview_num_frames)

        evaluator_template = config.get("EVALUATOR_SYSTEM_PROMPT", "")
        self.system_prompt = evaluator_template.format(
            overview_num_frames=overview_num_frames,
            skim_num_frames=skim_num_frames,
            focus_num_frames=focus_num_frames,
            max_inspect_frames=max_inspect_frames,
        )

    def construct_messages(
        self,
        question: str,
        duration: float,
        trajectory_steps: List[TrajectoryStep],
    ) -> List[dict]:
        history_blocks = []
        for step in trajectory_steps:
            action_dict = step.action.to_dict() if step.action else {}
            obs_text = str(step.observation)
            history_blocks.append(
                f"--- Step {step.step_id} ---\n"
                f"Actor Thought: {step.thought}\n"
                f"Action Called: {json.dumps(action_dict, ensure_ascii=False)}\n"
                f"Visual Observation:\n{obs_text}"
            )
        history_str = (
            "\n\n".join(history_blocks)
            if history_blocks
            else "No steps executed yet."
        )

        user_prompt = (
            f"Video Duration: {duration:.1f}s\n\n"
            f"Question:\n{question}\n\n"
            f"Actor's Trajectory History:\n{history_str}\n\n"
            "Please audit the trajectory and the latest visual observation following your Evaluation Workflow and Output Format."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def evaluate(
        self,
        question: str,
        duration: float,
        trajectory_steps: List[TrajectoryStep],
    ) -> str:
        messages = self.construct_messages(
            question=question,
            duration=duration,
            trajectory_steps=trajectory_steps,
        )
        response = call_llm_api(
            messages=messages,
            model_name=self.model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            api_version=self.api_version,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            seed=self.seed,
            temperature=self.temperature,
        )
        if response is None or not getattr(response, "choices", None):
            return (
                "- **Sufficiency Status**: `INSUFFICIENT`\n"
                "- **Evidence Audit**: Evaluator call failed; continue inspecting relevant segments.\n"
                "- **Reflexion Directive**: Call `inspect` on the most promising time window."
            )
        content = response.choices[0].message.content
        if not content or not str(content).strip():
            return (
                "- **Sufficiency Status**: `INSUFFICIENT`\n"
                "- **Evidence Audit**: Evaluator returned empty response; continue inspecting relevant segments.\n"
                "- **Reflexion Directive**: Call `inspect` on the most promising time window."
            )
        return content


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
        self.tool_names = list(tools)
        self.tools = self.tool_registry.resolve_tools(tools + ["answer"])
        self.output_dir = output_dir
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

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

        # Dual-Agent Reflexion Evaluator setup
        self.reflexion_enabled = bool(config.get("reflexion_enabled", True))
        self.evaluator: Optional[ReflexionEvaluatorAgent] = (
            ReflexionEvaluatorAgent(config=self.config)
            if self.reflexion_enabled and "EVALUATOR_SYSTEM_PROMPT" in self.config
            else None
        )

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

    def construct_initial_messages(self) -> List[dict]:
        """
        Build the initial [system, user] messages.
        Automatically selects `SYSTEM_PROMPT` (for unified `inspect` Actor)
        or `LEGACY_SYSTEM_PROMPT` (for legacy 3-tools Baseline when `inspect` is not used).
        """
        overview_num_frames = (
            self.config["frame_sampling_factor"] * self.config["overview_base"]
        )
        skim_num_frames = (
            self.config["frame_sampling_factor"] * self.config["skim_base"]
        )
        focus_num_frames = (
            self.config["frame_sampling_factor"] * self.config["focus_base"]
        )
        max_inspect_frames = self.config.get(
            "max_inspect_frames", overview_num_frames
        )

        use_legacy_prompt = (
            "inspect" not in getattr(self, "tool_names", ["inspect"])
            and "LEGACY_SYSTEM_PROMPT" in self.config
        )
        prompt_template = (
            self.config["LEGACY_SYSTEM_PROMPT"]
            if use_legacy_prompt
            else self.config["SYSTEM_PROMPT"]
        )

        system_prompt = prompt_template.format(
            overview_num_frames=overview_num_frames,
            skim_num_frames=skim_num_frames,
            focus_num_frames=focus_num_frames,
            max_inspect_frames=max_inspect_frames,
        )
        return [{"role": "system", "content": system_prompt}]

    def __parse_actions(self, thought: str) -> List[Action]:
        """
        Parse the actions from the thought.
        """
        actions = []

        try:
            parser_content = (
                f"Video Duration: {self.duration:.1f}s\n"
                f"Question: {self.question}\n\n"
                f"Please call the appropriate tool(s) based on the following thought:\n{thought}"
            )
            response = call_llm_api(
                messages=[
                    {
                        "role": "user",
                        "content": parser_content,
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
                            function_id=function_id or f"call_{function_name}_{tool_idx}",
                        )
                    )
        except Exception as e:
            print(f"Error parsing actions: {e}")
            return []

        non_answer_actions = [a for a in actions if a.function_name != "answer"]
        if non_answer_actions:
            actions = non_answer_actions
        elif len(actions) > 1:
            actions = [actions[0]]

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
            parameters.update({"vr": self.vr, "subtitles": self.subtitles})
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

    def run(self, question: str):
        self.reset()
        self.question = question
        subtitles_str = convert_to_free_form_text_representation(
            self.subtitles, content_type="subtitle"
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
            print("--------------------------------")

        for step in range(self.max_steps):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Step [{step + 1} / {self.max_steps}]: "
                        "Please follow the Thinking Policy to do **reasoning over the current state** "
                        "and **plan the next action(s) to take** following the Tool Calling Policy "
                        "and the Final Answer Policy. "
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
            thought = (
                response.choices[0].message.content
                if (response is not None and getattr(response, "choices", None))
                else ""
            )
            self.messages.append({"role": "assistant", "content": thought})
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] THOUGHT")
                print(thought)
                print("--------------------------------")

            ############################################################
            # ACTIONS
            ############################################################
            actions = self.__parse_actions(thought)
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
                                action.parameters, ensure_ascii=False
                            ),
                        },
                    }
                    for action in actions
                ]

            ############################################################
            # OBSERVATIONS & DUAL-AGENT REFLEXION EVALUATION
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

                traj_step = TrajectoryStep(
                    step_id=step + 1,
                    thought=thought,
                    action=action,
                    observation=observation,
                )
                self.trajectory_steps.append(traj_step)

                if action.function_name == "answer":
                    self.final_answer = observation.outcome
                    break

                reflection_text = None
                if self.reflexion_enabled and self.evaluator is not None:
                    try:
                        reflection_text = self.evaluator.evaluate(
                            question=question,
                            duration=self.duration,
                            trajectory_steps=self.trajectory_steps,
                        )
                    except Exception as e:
                        print(f"Evaluator error: {e}")
                        reflection_text = None

                    if reflection_text:
                        traj_step.reflection = reflection_text
                        if self.verbose:
                            print(
                                f"[STEP {step+1} / {self.max_steps}] EVALUATOR REFLEXION"
                            )
                            print(reflection_text)
                            print("--------------------------------")

                tool_msg_content = (
                    f"Observation from `{str(action.to_dict())}`:\n{outcome}"
                )
                if reflection_text:
                    tool_msg_content += (
                        f"\n\n[Reflexion Evaluator Feedback]:\n{reflection_text}"
                    )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id,
                        "content": tool_msg_content,
                    }
                )
            
            if len(actions) == 0:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "There is no function call in your response. YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.",
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
