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

"""3 - With agent lib and tool lib.

Using CaptainAgent with agent library and tool library. Based on: <https://docs.ag2.ai/latest/docs/user-guide/reference-agents/captainagent/#using-both-agent-library-and-tool-library>

Requirements: ag2[openai]==0.9.7, arxiv, chromadb, easyocr, huggingface-hub, markdownify, openai-whisper, pandas, pymupdf, python-pptx, scipy, sentence-transformers, wikipedia-api
Tags: CaptainAgent, ag2
🧩 generated with ❤️ by Waldiez.
"""


# Imports

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
from autogen.agentchat.contrib.captainagent import CaptainAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.events import BaseEvent
from autogen.io.run_response import AsyncRunResponseProtocol, RunResponseProtocol
import numpy as np

# Common environment variable setup for Waldiez flows
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
# This section assumes that a file named "wf_3___with_agent_lib_a_api_keys"
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


__MODELS_MODULE__ = load_api_key_module("wf_3___with_agent_lib_a")


def get_wf_3___with_agent_lib_a_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_wf_3___with_agent_lib_a_model_api_key(model_name)


# Tools

# Load tool secrets module if needed
# NOTE:
# This section assumes that a file named "wf_3___with_agent_lib_a_new_tool_secrets"
# exists in the same directory as this file.
# This file contains the secrets for the tool used in this flow.
# It should be .gitignored and not shared publicly.
# If this file is not present, you can either create it manually
# or change the way secrets are loaded in the flow.


def load_tool_secrets_module(flow_name: str, tool_name: str) -> ModuleType:
    """Load the tool secrets module for the given flow name and tool name.

    Parameters
    ----------
    flow_name : str
        The flow name.

    Returns
    -------
    ModuleType
        The loaded module.
    """
    module_name = f"{flow_name}_{tool_name}_secrets"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


load_tool_secrets_module("wf_3___with_agent_lib_a", "new_tool")

"""Replace this with your code.

Add any code here that will be placed at the top of the whole flow.
"""

# Example:
# global variable
# DATABASE = {
#     "users": [
#         {"id": 1, "name": "Alice"},
#         {"id": 2, "name": "Bob"},
#     ],
#     "posts": [
#         {"id": 1, "title": "Hello, world!", "author_id": 1},
#         {"id": 2, "title": "Another post", "author_id": 2},
#     ],
# }
#
# Add your code below
RAPID_API_KEY = os.environ["RAPID_API_KEY"]

# Models

gpt_4o_llm_config: dict[str, Any] = {
    "model": "gpt-4o",
    "api_type": "openai",
    "api_key": get_wf_3___with_agent_lib_a_model_api_key("gpt_4o"),
}

# Agents

captain_executor = LocalCommandLineCodeExecutor(
    work_dir="groupchat",
)

captain = CaptainAgent(
    name="captain",
    description="A new Captain agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config={"executor": captain_executor},
    is_termination_msg=None,  # pyright: ignore
    agent_config_save_path=os.getcwd(),
    agent_lib="captain_agent_lib.json",
    tool_lib="default",
    nested_config={
        "autobuild_init_config": {
            "config_file_or_env": "captain_llm_config.json",
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 2048},
            "code_execution_config": {
                "timeout": 300,
                "work_dir": "groupchat",
                "last_n_messages": 1,
                "use_docker": False,
            },
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": None,
        "max_turns": 5,
    },
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_llm_config,
        ],
        cache_seed=None,
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


def callable_message_gettranscript(
    sender: ConversableAgent,
    recipient: ConversableAgent,
    context: dict[str, Any],
) -> Union[dict[str, Any], str]:
    query = """# Task
    Your task is to solve a question given by a user.

    # Question
    Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.

    What does Teal'c say in response to the question "Isn't that hot?"
    """.strip()
    return query


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
    data = [dict(zip(column_names, row)) for row in rows]
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


def main(on_event: Optional[Callable[[BaseEvent], bool]] = None) -> RunResponseProtocol:
    """Start chatting.

    Returns
    -------
    RunResponseProtocol
        The result of the chat session, which can be a single ChatResult,
        a list of ChatResults, or a dictionary mapping integers to ChatResults.

    Raises
    ------
    RuntimeError
        If the chat session fails.
    """
    results = user_proxy.run(
        captain,
        summary_method="last_msg",
        max_turns=1,
        clear_history=True,
        message=callable_message_gettranscript,
    )
    if on_event:
        if not isinstance(results, list):
            results = [results]
        for index, result in enumerate(results):
            for event in result.events:
                try:
                    should_continue = on_event(event)
                except Exception as e:
                    raise RuntimeError("Error in event handler: " + str(e)) from e
                if event.type == "run_completion":
                    should_continue = False
                if not should_continue:
                    break
    else:
        if not isinstance(results, list):
            results = [results]
        for result in results:
            result.process()

    stop_logging()
    return results


def call_main() -> None:
    """Run the main function and print the results."""
    results: list[RunResponseProtocol] = main()
    results_dicts: list[dict[str, Any]] = []
    for result in results:
        result_summary = result.summary
        result_messages = result.messages
        result_cost = result.cost
        cost: dict[str, Any] | None = None
        if result_cost:
            cost = result_cost.model_dump(mode="json", fallback=str)
        results_dicts.append(
            {
                "summary": result_summary,
                "messages": result_messages,
                "cost": cost,
            }
        )

    results_dict = {
        "results": results_dicts,
    }
    print(json.dumps(results_dict, indent=2))


if __name__ == "__main__":
    # Let's go!
    call_main()
