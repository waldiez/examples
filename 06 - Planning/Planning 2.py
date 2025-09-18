#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0.
# Copyright (c) 2024 - 2025 Waldiez and contributors.
# flake8: noqa: E501

# pylint: disable=line-too-long,unknown-option-value,unused-argument,unused-import,unused-variable,invalid-name
# pylint: disable=import-error,import-outside-toplevel,inconsistent-quotes,missing-function-docstring,missing-param-doc,missing-return-doc
# pylint: disable=ungrouped-imports,unnecessary-lambda-assignment,too-many-arguments,too-many-locals,too-many-try-statements,broad-exception-caught

# type: ignore

# pyright: reportUnusedImport=false,reportMissingTypeStubs=false,reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false,reportUnknownLambdaType=false,reportUnnecessaryIsInstance=false
# pyright: reportUnknownVariableType=false

"""Planning 2.

Planning and Stock Report Generation, using Ag2’s group patterns.

Requirements: ag2[openai]==0.9.9
Tags: Planning, Stock report, Group
🧩 generated with ❤️ by Waldiez.
"""


# Imports

import asyncio
import csv
import importlib
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from pprint import pprint
from types import ModuleType
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import autogen  # type: ignore
from autogen import (
    Agent,
    Cache,
    ChatResult,
    ConversableAgent,
    GroupChat,
    UserProxyAgent,
    runtime_logging,
)
from autogen.agentchat import GroupChatManager, run_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.events import BaseEvent
from autogen.io.run_response import AsyncRunResponseProtocol, RunResponseProtocol
import numpy as np
from dotenv import load_dotenv

# Common environment variable setup for Waldiez flows
load_dotenv(override=True)
os.environ["AUTOGEN_USE_DOCKER"] = "0"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
#
# let's try to avoid:
# module 'numpy' has no attribute '_no_nep50_warning'"
# ref: https://github.com/numpy/numpy/blob/v2.2.2/doc/source/release/2.2.0-notes.rst#nep-50-promotion-state-option-removed
os.environ["NEP50_DEPRECATION_WARNING"] = "0"
os.environ["NEP50_DISABLE_WARNING"] = "1"
os.environ["NPY_PROMOTION_STATE"] = "weak"
if not hasattr(np, "_no_pep50_warning"):

    import contextlib
    from typing import Generator

    @contextlib.contextmanager
    def _np_no_nep50_warning() -> Generator[None, None, None]:
        """Dummy function to avoid the warning.

        Yields
        ------
        None
            Nothing.
        """
        yield

    setattr(np, "_no_pep50_warning", _np_no_nep50_warning)  # noqa

# Start logging.


def start_logging() -> None:
    """Start logging."""
    runtime_logging.start(
        logger_type="sqlite",
        config={"dbname": "flow.db"},
    )


start_logging()

# Load model API keys
# NOTE:
# This section assumes that a file named:
# "planning_2_api_keys.py"
# exists in the same directory as this file.
# This file contains the API keys for the models used in this flow.
# It should be .gitignored and not shared publicly.
# If this file is not present, you can either create it manually
# or change the way API keys are loaded in the flow.


def load_api_key_module(flow_name: str) -> ModuleType:
    """Load the api key module.

    Parameters
    ----------
    flow_name : str
        The flow name.

    Returns
    -------
    ModuleType
        The api keys loading module.
    """
    module_name = f"{flow_name}_api_keys"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


__MODELS_MODULE__ = load_api_key_module("planning_2")


def get_planning_2_model_api_key(model_name: str) -> str:
    """Get the model api key.
    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    str
        The model api key.
    """
    return __MODELS_MODULE__.get_planning_2_model_api_key(model_name)


# Models

gpt_4_turbo_llm_config: dict[str, Any] = {
    "model": "gpt-4-turbo",
    "api_type": "openai",
    "api_key": get_planning_2_model_api_key("gpt_4_turbo"),
}

# Agents

executor_executor = LocalCommandLineCodeExecutor(
    work_dir="coding",
    timeout=60,
)

engineer = ConversableAgent(
    name="engineer",
    description="An engineer that writes code based on the plan provided by the planner.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

executor = ConversableAgent(
    name="executor",
    description="Executor agent",
    system_message="Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config={"executor": executor_executor},
    is_termination_msg=None,  # pyright: ignore
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=False,  # pyright: ignore
)

planner = ConversableAgent(
    name="planner",
    description="Planner agent",
    system_message="Given a task, please determine what information is needed to complete the task. Please note that the information will all be retrieved using Python code. Please only suggest information that can be retrieved using Python code. After each step is done by others, check the progress and instruct the remaining steps. If a step fails, try to workaround.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    description="A new User proxy agent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    llm_config=False,  # pyright: ignore
)

writer = ConversableAgent(
    name="writer",
    description="Writer agent",
    system_message="Writer. Please write blogs in markdown format (with relevant titles) and put the content in pseudo ```md``` code block. You take feedback from the admin and refine your blog.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

manager_pattern = AutoPattern(
    initial_agent=planner,
    agents=[planner, engineer, executor, writer],
    user_agent=user_proxy,
    group_manager_args={
        "llm_config": autogen.LLMConfig(
            config_list=[
                gpt_4_turbo_llm_config,
            ],
            cache_seed=None,
        ),
        "name": "manager",
    },
)


def get_sqlite_out(dbname: str, table: str, csv_file: str) -> None:
    """Convert a sqlite table to csv and json files.

    Parameters
    ----------
    dbname : str
        The sqlite database name.
    table : str
        The table name.
    csv_file : str
        The csv file name.
    """
    conn = sqlite3.connect(dbname)
    query = f"SELECT * FROM {table}"  # nosec
    try:
        cursor = conn.execute(query)
    except sqlite3.OperationalError:
        conn.close()
        return
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row, strict=True)) for row in rows]
    conn.close()
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=column_names)
        csv_writer.writeheader()
        csv_writer.writerows(data)
    json_file = csv_file.replace(".csv", ".json")
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def stop_logging() -> None:
    """Stop logging."""
    runtime_logging.stop()
    if not os.path.exists("logs"):
        os.makedirs("logs")
    for table in [
        "chat_completions",
        "agents",
        "oai_wrappers",
        "oai_clients",
        "version",
        "events",
        "function_calls",
    ]:
        dest = os.path.join("logs", f"{table}.csv")
        get_sqlite_out("flow.db", table, dest)


# Start chatting


def main(
    on_event: Optional[Callable[[BaseEvent], bool]] = None,
) -> list[dict[str, Any]]:
    """Start chatting.

    Returns
    -------
    list[dict[str, Any]]
        The result of the chat session.

    Raises
    ------
    SystemExit
        If the user interrupts the chat session.
    """
    results: list[RunResponseProtocol] | RunResponseProtocol = []
    result_dicts: list[dict[str, Any]] = []
    with Cache.disk(cache_seed=42) as cache:  # pyright: ignore
        results = run_group_chat(
            pattern=manager_pattern,
            messages="Write a blogpost about the stock price performance of Nvidia in the past month.",
            max_rounds=20,
        )
        if not isinstance(results, list):
            results = [results]  # pylint: disable=redefined-variable-type
        if on_event:
            for index, result in enumerate(results):
                for event in result.events:
                    try:
                        should_continue = on_event(event)
                    except BaseException as e:
                        stop_logging()
                        print(f"Error in event handler: {e}")
                        raise SystemExit("Error in event handler: " + str(e)) from e
                    if event.type == "run_completion":
                        break
                    if not should_continue:
                        stop_logging()
                        raise SystemExit("Event handler stopped processing")
                result_dict = {
                    "index": index,
                    "messages": result.messages,
                    "summary": result.summary,
                    "cost": (
                        result.cost.model_dump(mode="json", fallback=str)
                        if result.cost
                        else None
                    ),
                    "context_variables": (
                        result.context_variables.model_dump(mode="json", fallback=str)
                        if result.context_variables
                        else None
                    ),
                    "last_speaker": result.last_speaker,
                    "uuid": str(result.uuid),
                }
                result_dicts.append(result_dict)
        else:
            for index, result in enumerate(results):
                result.process()
                result_dict = {
                    "index": index,
                    "messages": result.messages,
                    "summary": result.summary,
                    "cost": (
                        result.cost.model_dump(mode="json", fallback=str)
                        if result.cost
                        else None
                    ),
                    "context_variables": (
                        result.context_variables.model_dump(mode="json", fallback=str)
                        if result.context_variables
                        else None
                    ),
                    "last_speaker": result.last_speaker,
                    "uuid": str(result.uuid),
                }
                result_dicts.append(result_dict)

        stop_logging()
    return result_dicts


def call_main() -> None:
    """Run the main function and print the results."""
    results: list[dict[str, Any]] = main()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Let's go!
    call_main()
