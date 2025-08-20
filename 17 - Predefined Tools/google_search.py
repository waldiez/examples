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

"""google search flow.

A example waldiez flow using google search

Requirements: ag2[google-search,gemini], ag2[openai]==0.9.9
Tags: websearch
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
    AssistantAgent,
    Cache,
    ChatResult,
    ConversableAgent,
    GroupChat,
    UserProxyAgent,
    register_function,
    runtime_logging,
)
from autogen.events import BaseEvent
from autogen.io.run_response import AsyncRunResponseProtocol, RunResponseProtocol
from autogen.tools.experimental import GoogleSearchTool
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
# "google_search_flow_api_keys.py"
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


__MODELS_MODULE__ = load_api_key_module("google_search_flow")


def get_google_search_flow_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_google_search_flow_model_api_key(model_name)


# Tools

# Load tool secrets module if needed
# NOTE:
# This section assumes that a file named:
# "google_search_flow_google_search_secrets.py"
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


load_tool_secrets_module("google_search_flow", "google_search")


def google_search(
    query: str,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Perform a Google search and return formatted results.

    Args:
        query: The search query string.
        num_results: The maximum number of results to return. Defaults to 10.
    Returns:
        A list of dictionaries of the search results.
    """
    google_search_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
    if not google_search_api_key:
        raise ValueError("GOOGLE_SEARCH_API_KEY is required for Google search tool.")
    google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")
    if not google_search_engine_id:
        raise ValueError("Google Search Engine ID is required for Google search tool.")
    google_search_tool = GoogleSearchTool(
        search_api_key=google_search_api_key,
        search_engine_id=google_search_engine_id,
    )
    return google_search_tool(
        query=query,
        search_api_key=google_search_api_key,
        search_engine_id=google_search_engine_id,
        num_results=num_results,
    )


# Models

gpt_4_1_llm_config: dict[str, Any] = {
    "model": "gpt-4.1",
    "api_type": "openai",
    "api_key": get_google_search_flow_model_api_key("gpt_4_1"),
}

# Agents

assistant = AssistantAgent(
    name="assistant",
    description="A new Assistant agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=42,
    ),
)

user = UserProxyAgent(
    name="user",
    description="A new User agent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    llm_config=False,  # pyright: ignore
)

register_function(
    google_search,
    caller=assistant,
    executor=user,
    name="google_search",
    description="Search Google for a given query.",
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
        results = user.run(
            assistant,
            cache=cache,
            summary_method="last_msg",
            max_turns=5,
            clear_history=True,
            message="Tell me a few things about what are the patterns in groupchat in AG2",
        )
        if on_event:
            if not isinstance(results, list):
                results = [results]  # pylint: disable=redefined-variable-type
            for index, result in enumerate(results):
                for event in result.events:
                    try:
                        should_continue = on_event(event)
                    except BaseException as e:
                        print(f"Error in event handler: {e}")
                        raise SystemExit("Error in event handler: " + str(e)) from e
                    if event.type == "run_completion":
                        break
                    if not should_continue:
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
            if not isinstance(results, list):
                results = [results]  # pylint: disable=redefined-variable-type
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
