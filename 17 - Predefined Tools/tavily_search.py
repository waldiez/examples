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

"""A example waldiez flow using tavily search.

A example waldiez flow using tavily search

Requirements: ag2[openai]==0.9.7, ag2[tavily, openai]
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
from autogen.tools.experimental import TavilySearchTool
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
# This section assumes that a file named "a_example_waldiez_fl_api_keys"
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


__MODELS_MODULE__ = load_api_key_module("a_example_waldiez_fl")


def get_a_example_waldiez_fl_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_a_example_waldiez_fl_model_api_key(model_name)


# Tools

# Load tool secrets module if needed
# NOTE:
# This section assumes that a file named "a_example_waldiez_fl_tavily_search_secrets"
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


load_tool_secrets_module("a_example_waldiez_fl", "tavily_search")


def tavily_search(
    query: str,
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", ""),
    search_depth: str = "basic",
    topic: str = "general",
    include_answer: str = "basic",
    include_raw_content: bool = False,
    include_domains: list[str] = [],
    num_results: int = 5,
) -> list[dict[str, Any]]:
    """Performs a search using the Tavily API and returns formatted results.

    Args:
        query: The search query string.
        tavily_api_key: The API key for Tavily (injected dependency).
        search_depth: The depth of the search ('basic' or 'advanced'). Defaults to "basic".
        include_answer: Whether to include an AI-generated answer ('basic' or 'advanced'). Defaults to "basic".
        include_raw_content: Whether to include raw content in the results. Defaults to False.
        include_domains: A list of domains to include in the search. Defaults to [].
        num_results: The maximum number of results to return. Defaults to 5.

    Returns:
        A list of dictionaries, each containing 'title', 'link', and 'snippet' of a search result.

    Raises:
        ValueError: If the Tavily API key is not available.
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is required for Tavily search tool.")
    tavily_search_tool = TavilySearchTool(
        tavily_api_key=tavily_api_key,
    )
    return tavily_search_tool(
        query=query,
        tavily_api_key=tavily_api_key,
        search_depth=search_depth,
        topic=topic,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        num_results=num_results,
    )


# Models

gpt_4_1_llm_config: dict[str, Any] = {
    "model": "gpt-4.1",
    "api_type": "openai",
    "api_key": get_a_example_waldiez_fl_model_api_key("gpt_4_1"),
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
    tavily_search,
    caller=assistant,
    executor=user,
    name="tavily_search",
    description="Search Tavily for a given query.",
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
    with Cache.disk(cache_seed=42) as cache:  # pyright: ignore
        results = user.run(
            assistant,
            cache=cache,
            summary_method="last_msg",
            max_turns=4,
            clear_history=True,
            message="Who won the nba title in 2025?",
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
