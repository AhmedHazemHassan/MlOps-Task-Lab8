"""AutoGen-based Planner/Executor multi-agent workflow.

Pattern
-------
We use a **UserProxy + GroupChat Manager/Worker pattern**:

- ``UserProxyAgent`` ("user") represents the human driving the task.
- ``AssistantAgent`` ("planner") decomposes the goal into an ordered plan.
- ``AssistantAgent`` ("executor") simulates executing each step and logging it.
- ``GroupChatManager`` ("supervisor") coordinates planner and executor turns.

A simple tool, ``save_plan_to_file``, is registered for execution so that the
agents can persist the final plan to disk when appropriate.
"""

from __future__ import annotations

from pprint import pprint
from typing import Tuple

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

from config import get_llm_config
from tools import save_plan_to_file


def create_planner_executor_agents() -> Tuple[UserProxyAgent, GroupChatManager]:
    """Create and wire up AutoGen agents for the Planner/Executor workflow.

    Returns
    -------
    (user_proxy, manager):
        The user-facing proxy agent and the group chat manager that
        coordinates planner and executor.
    """

    llm_config = get_llm_config()

    # User proxy that drives the conversation. We disable arbitrary code
    # execution for safety and keep the interaction fully automated.
    user_proxy = UserProxyAgent(
        name="user",
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    # Register the ``save_plan_to_file`` tool so that assistants can call it.
    @user_proxy.register_for_execution()
    def save_plan_tool(steps: list[str]) -> str:  # type: ignore[override]
        """Tool: save the current plan steps to a log file."""

        return save_plan_to_file(steps)

    planner = AssistantAgent(
        name="planner",
        llm_config=llm_config,
        system_message=(
            "You are an MLOps planning assistant. Given a high-level goal, "
            "produce a numbered list of 4-8 concrete steps to achieve it. "
            "Focus on practical, implementation-oriented actions."
        ),
    )

    executor = AssistantAgent(
        name="executor",
        llm_config=llm_config,
        system_message=(
            "You are an MLOps execution assistant. For each step in the "
            "plan, describe briefly how to carry it out in practice, then "
            "call `save_plan_tool` once you have the final plan ready."
        ),
    )

    groupchat = GroupChat(
        agents=[user_proxy, planner, executor],
        messages=[],
        max_round=10,
    )

    manager = GroupChatManager(
        name="supervisor",
        groupchat=groupchat,
        llm_config=llm_config,
        system_message=(
            "You are a supervisor that coordinates a planner and an executor "
            "agent. Ensure the planner proposes a reasonable plan first, "
            "then have the executor elaborate on each step and call tools "
            "when needed. Stop the conversation when a clear plan and "
            "execution notes have been produced."
        ),
    )

    return user_proxy, manager


def main() -> None:
    """Run an end-to-end Planner/Executor conversation using AutoGen."""

    user_proxy, manager = create_planner_executor_agents()

    print("=== AutoGen Planner/Executor Example ===")
    user_goal = input("Goal: ")
    print(f"User goal: {user_goal}\n")

    # Initiate the group chat via the user proxy. The manager will coordinate
    # between planner and executor until the conversation is complete.
    chat_result = user_proxy.initiate_chat(
        manager,
        message=user_goal,
    )

    print("\n=== Final Conversation Result ===")
    pprint(chat_result)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
