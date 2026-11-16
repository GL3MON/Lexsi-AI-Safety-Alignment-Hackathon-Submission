"""Isolated tau2 solo Gym rollouts and real terminal rewards for TRL."""

import json
import math
import threading
import warnings
from uuid import uuid4

from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
from tau2.gym.gym_agent import AgentGymEnv


def configure(cfg, run, allowed_ids, get_tokenizer, get_template):
    """Bind this single-process rollout bridge to the active experiment."""
    global CFG, RUN, ALLOWED_IDS, PHASE, _get_tokenizer, _get_template
    CFG, RUN = cfg, run
    ALLOWED_IDS = set(allowed_ids)
    PHASE = "cpu_smoke"
    EVENTS.clear()
    _get_tokenizer, _get_template = get_tokenizer, get_template


# These globals describe the CURRENT phase, not a reward label provided to the model.
# This notebook is intentionally single-process/single-GPU so this association is explicit.
PHASE = "unconfigured"
ALLOWED_IDS = set()
EVENTS = []
EVENT_LOCK = threading.Lock()


def _get_tokenizer():
    return None


def _get_template():
    return None


def set_phase(name, allowed_ids):
    """Declare which tasks the harness is permitted to execute in the next phase."""
    global PHASE, ALLOWED_IDS
    PHASE = name
    ALLOWED_IDS = set(allowed_ids)


def append_event(event):
    """Persist a compact episode record without retaining all trajectories in RAM."""
    with EVENT_LOCK:
        EVENTS.append(event)
        with (RUN / "episodes.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")


class Tau2SoloRollout:
    """One isolated τ-bench Gym episode per TRL completion; no user simulator."""

    def __init__(self):
        # Instantiation is cheap: no thread or domain database is created until reset.
        self._env = None
        self._final = None
        self._gym_done = False
        self._actions = []
        self._task_id = None
        self._phase = None
        self._last_info = None

    def reset(self, task_id: str, **kwargs) -> str:
        """Start the task selected by the dataset, restoring a fresh scenario state.

        TRL forwards dataset fields, including prompt and task_id, as keyword arguments.
        The returned context is appended to the dataset's last user message. Nothing
        in the returned text contains task evaluation criteria or reference actions.
        """
        if task_id not in ALLOWED_IDS:
            raise ValueError(f"Task {task_id} is not allowed during phase {PHASE}.")
        if self._env is not None:
            self._finish()  # Defensive cleanup before replacing an existing session.
        self._final = None
        self._gym_done = False
        self._actions = []
        self._task_id = task_id
        self._phase = PHASE
        self._rollout_id = uuid4().hex
        self._env = AgentGymEnv(
            domain=CFG.domain,
            task_id=task_id,
            solo_mode=True,
            max_steps=2 * CFG.max_actions + 4,
            communicate_scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING),
        )
        try:
            observation, info = self._env.reset(seed=CFG.seed)
            self._last_info = info
            if observation:
                raise RuntimeError("Expected an empty initial solo-mode observation.")
            catalog = [tool.openai_schema for tool in info["tools"]]
            context = (
                "\n\n<ticket>\n"
                + info["task"].ticket
                + "\n</ticket>\n"
                + "<policy>\n"
                + info["policy"]
                + "\n</policy>\n"
                + "<gym_tools>\n"
                + json.dumps(catalog)
                + "\n</gym_tools>\n"
                + "Use the execute tool to call these Gym tools. Pass the tool name and a JSON object "
                "encoded as arguments_json. Use one action at a time and inspect its observation. "
                "When finished call execute(tool_name='done', arguments_json='{}'), then stop."
            )
            tokenizer = _get_tokenizer()
            template = _get_template()
            if tokenizer is not None:
                # Fail before model generation if the unmodified policy/ticket is too long.
                # Do not fix a context overflow by silently deleting policy requirements.
                from transformers.utils import get_json_schema

                messages = [{"role": "user", "content": BASE_PROMPT + context}]
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tools=[get_json_schema(self.execute)],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    **({"chat_template": template} if template else {}),
                )
                if len(token_ids) + CFG.max_completion_tokens > CFG.context_limit:
                    raise ValueError(
                        f"Prompt has {len(token_ids)} tokens; completion reserve is "
                        f"{CFG.max_completion_tokens}; context limit is {CFG.context_limit}. "
                        "Choose a larger supported context or reduce the completion budget."
                    )
            return context
        except Exception:
            # Do not leave the worker waiting if prompt construction/preflight fails.
            self._finish()
            raise

    def execute(self, tool_name: str, arguments_json: str = "{}") -> str:
        """Execute one τ-bench tool and return its observation.

        Args:
            tool_name: Name from the Gym tool catalog, including done to finish.
            arguments_json: JSON object encoded as a string containing keyword arguments.

        Returns:
            The actual Gym observation and terminal flag as a JSON string. This output
            is environment feedback, not model text to imitate during optimization.
        """
        if self._env is None or self._final is not None:
            return json.dumps(
                {
                    "terminated": True,
                    "message": "Episode is already finished; stop calling tools.",
                }
            )
        if self._gym_done:
            return json.dumps(
                {
                    "terminated": True,
                    "message": "Episode finished; emit a final response.",
                }
            )
        if len(self._actions) >= CFG.max_actions:
            self._finish()
            return json.dumps(
                {"terminated": True, "message": "External action budget exhausted."}
            )
        try:
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError("arguments_json must encode an object.")
        except (ValueError, TypeError) as error:
            # A policy-format error is an observation the model can recover from.
            # It consumes an action but is not a successful environment action.
            self._actions.append({"tool_name": tool_name, "invalid_json": str(error)})
            return json.dumps({"error": str(error), "terminated": False})
        action = {"name": tool_name, "arguments": arguments}
        observation, reward, terminated, truncated, info = self._env.step(
            json.dumps(action)
        )
        self._actions.append({"action": action, "observation": observation})
        self._last_info = info
        self._gym_done = bool(terminated or truncated)
        return json.dumps({"observation": observation, "terminated": self._gym_done})

    def _finish(self):
        """End the episode, read τ-bench's evaluated reward, save it once, then clean up."""
        if self._final is not None:
            return self._final
        if self._env is None:
            raise RuntimeError("Cannot score an environment that was never reset.")
        forced_stop = not self._gym_done
        try:
            if not self._gym_done:
                _, _, terminated, truncated, info = self._env.step("done()")
                self._gym_done = bool(terminated or truncated)
                self._last_info = info
            if not self._gym_done:
                raise RuntimeError("Gym did not terminate after done().")
            simulation = json.loads(self._last_info["simulation_run"])
            if not simulation.get("reward_info"):
                # Gym's orchestrator catches some exceptions internally. Missing
                # evaluation is an infrastructure error, not a legitimate failed sample.
                raise RuntimeError(
                    "Missing evaluated simulation. Inspect the Gym error log."
                )
            task_reward = float(simulation["reward_info"]["reward"])
            if not math.isfinite(task_reward):
                raise RuntimeError("Non-finite task reward.")
            path = RUN / f"episode-{self._rollout_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "phase": self._phase,
                        "actions": self._actions,
                        "forced_stop": forced_stop,
                        "simulation": simulation,
                    },
                    indent=2,
                )
            )
            event = {
                "phase": self._phase,
                "task_id": self._task_id,
                "rollout_id": self._rollout_id,
                "task_reward": task_reward,
                "actions": len(self._actions),
                "forced_stop": forced_stop,
                "simulation_path": str(path),
            }
            append_event(event)
            self._final = event
            return event
        finally:
            self._env.close()
            self._env = None


def terminal_task_reward(completions, environments, task_id, **kwargs):
    """Return one evaluated τ-bench terminal reward per on-policy completion.

    TRL 1.7.1 forwards the per-completion environment list to synchronous reward
    callables. These environments already contain the states reached by the sampled
    policy. We finish them and read their actual rewards; we never replay an action
    against another sample's state or substitute a tool-validity score.
    """
    if not (len(completions) == len(environments) == len(task_id)):
        raise RuntimeError("GRPO environment/completion/task alignment mismatch.")
    rewards = []
    for index, (assigned_id, environment) in enumerate(zip(task_id, environments, strict=True)):
        if assigned_id != environment._task_id:
            raise RuntimeError("A rollout was routed to the wrong task.")
        # Persist the model conversation, including tool errors caught by TRL,
        # before finishing Gym. The Gym transcript alone omits rejected calls.
        tokenizer = _get_tokenizer()
        ids = kwargs.get("completion_ids")
        raw_text = (
            tokenizer.decode(ids[index], skip_special_tokens=False)
            if tokenizer is not None and ids is not None else None
        )
        record = {
            "phase": environment._phase,
            "task_id": assigned_id,
            "rollout_id": environment._rollout_id,
            "messages": completions[index],
            "raw_text": raw_text,
        }
        stem = RUN / f"completion-{environment._rollout_id}"
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2))
        stem.with_suffix(".txt").write_text(
            f"Phase: {environment._phase}\nTask: {assigned_id}\n\n"
            + (raw_text if raw_text is not None else json.dumps(completions[index], indent=2))
        )
        print(f"Model output: {stem.with_suffix('.txt')}", flush=True)
        rewards.append(environment._finish()["task_reward"])
    return rewards


def cleanup_trainer_environments(trainer):
    """Finish outstanding Gym episodes after a generation/training exception."""
    errors = []
    for environment in trainer.environments:
        if environment._env is not None:
            try:
                environment._finish()
            except Exception as error:
                errors.append(str(error))
    if errors:
        warnings.warn("Environment cleanup errors: " + "; ".join(errors))


BASE_PROMPT = (
    "Resolve the assigned telecom ticket independently in solo mode. "
    "Follow the policy, ground decisions in tool observations, and do not invent results. "
    "Use the execute tool for Gym actions. The task context follows."
)
