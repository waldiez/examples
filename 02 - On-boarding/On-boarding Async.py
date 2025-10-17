#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0.
# Copyright (c) 2024 - 2025 Waldiez and contributors.
# flake8: noqa: E501

# pylint: disable=broad-exception-caught,f-string-without-interpolation,invalid-name,import-error,import-outside-toplevel,inconsistent-quotes,line-too-long,missing-function-docstring
# pylint: disable=missing-param-doc,missing-return-doc,no-member,pointless-string-statement,too-complex,too-many-arguments,too-many-locals,too-many-try-statements
# pylint: disable=ungrouped-imports,unnecessary-lambda-assignment,unknown-option-value,unused-argument,unused-import,unused-variable

# type: ignore

# pyright: reportArgumentType=false,reportAttributeAccessIssue=false,reportCallInDefaultInitializer=false,reportDeprecated=false,reportDuplicateImport=false,reportMissingTypeStubs=false
# pyright: reportOperatorIssue=false,reportOptionalMemberAccess=false,reportPossiblyUnboundVariable=false,reportUnreachable=false,reportUnusedImport=false,reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false,reportUnknownLambdaType=false,reportUnnecessaryIsInstance=false,reportUnusedParameter=false,reportUnusedVariable=false,reportUnknownVariableType=false

"""On-boarding Async.

Async version of Sequential Chats and Customer Onboarding

Requirements: ag2[anthropic]==0.9.10, ag2[openai]==0.9.10
Tags: Sequential, Customer, On-boarding, Onboarding
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
import traceback
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

import aiofiles
import aiosqlite
import anyio
import nest_asyncio
from aiocsv import AsyncDictWriter

import autogen  # type: ignore
from autogen import (
    Agent,
    AssistantAgent,
    Cache,
    ChatResult,
    ConversableAgent,
    GroupChat,
    UserProxyAgent,
    runtime_logging,
)
from autogen.events import BaseEvent
from autogen.io.run_response import AsyncRunResponseProtocol, RunResponseProtocol
import numpy as np
from dotenv import load_dotenv

# pylint: disable=broad-exception-caught
try:
    nest_asyncio.apply()
except BaseException:
    pass  # maybe on uvloop?
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
    # pylint: disable=import-outside-toplevel
    from anyio.from_thread import start_blocking_portal

    with start_blocking_portal(backend="asyncio") as portal:
        portal.call(
            runtime_logging.start,
            None,
            "sqlite",
            {"dbname": "flow.db"},
        )


start_logging()

# Load model API keys
# NOTE:
# This section assumes that a file named:
# "on_boarding_async_api_keys.py"
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


__MODELS_MODULE__ = load_api_key_module("on_boarding_async")


def get_on_boarding_async_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_on_boarding_async_model_api_key(model_name)


# Models

claude_3_7_sonnet_20250219_llm_config: dict[str, Any] = {
    "model": "claude-3-7-sonnet-20250219",
    "api_type": "anthropic",
    "api_key": get_on_boarding_async_model_api_key("claude_3_7_sonnet_20250219"),
}

gpt_3_5_turbo_llm_config: dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "api_type": "openai",
    "api_key": get_on_boarding_async_model_api_key("gpt_3_5_turbo"),
}

# Agents

customer_engagement_agent = AssistantAgent(
    name="customer_engagement_agent",
    description="A customer_engagement_agent agent.",
    system_message="You are a helpful customer service agent here to provide fun for the customer based on the user's personal information and topic preferences. This could include fun facts, jokes, or interesting stories. Make sure to make it engaging and fun! Return 'TERMINATE' when you are done.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "") == keyword
        for keyword in ["TERMINATE"]
    ),
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=None,
    ),
)

customer_proxy = UserProxyAgent(
    name="customer_proxy",
    description="A new User proxy agent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=False,
)

personal_information_agent = AssistantAgent(
    name="personal_information_agent",
    description="A customer on-boarding agent.",
    system_message="You are a helpful customer on-boarding agent, you are here to help new customers get started with our product. Your job is to gather customer's name and location. Do not ask for other information. Return 'TERMINATE' when you have gathered all the information.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "") == keyword
        for keyword in ["TERMINATE"]
    ),
    llm_config=autogen.LLMConfig(
        config_list=[
            claude_3_7_sonnet_20250219_llm_config,
        ],
        cache_seed=None,
    ),
)

topic_preference_agent = AssistantAgent(
    name="topic_preference_agent",
    description="A topic preference agent",
    system_message="You are a helpful customer service agent here to provide fun for the customer based on the user's personal information and topic preferences. This could include fun facts, jokes, or interesting stories. Make sure to make it engaging and fun! Return 'TERMINATE' when you are done.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "") == keyword
        for keyword in ["TERMINATE"]
    ),
    llm_config=autogen.LLMConfig(
        config_list=[
            claude_3_7_sonnet_20250219_llm_config,
        ],
        cache_seed=None,
    ),
)


async def get_sqlite_out(dbname: str, table: str, csv_file: str) -> None:
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
    conn = await aiosqlite.connect(dbname)
    query = f"SELECT * FROM {table}"  # nosec
    try:
        cursor = await conn.execute(query)
    except BaseException:  # pylint: disable=broad-exception-caught
        await conn.close()
        return
    rows = await cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row, strict=True)) for row in rows]
    await cursor.close()
    await conn.close()
    async with aiofiles.open(csv_file, "w", newline="", encoding="utf-8") as file:
        csv_writer = AsyncDictWriter(file, fieldnames=column_names, dialect="unix")
        await csv_writer.writeheader()
        await csv_writer.writerows(data)
    json_file = csv_file.replace(".csv", ".json")
    async with aiofiles.open(json_file, "w", encoding="utf-8", newline="\n") as file:
        await file.write(json.dumps(data, indent=4, ensure_ascii=False))


async def stop_logging() -> None:
    """Stop logging."""
    await asyncio.to_thread(runtime_logging.stop)
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
        await get_sqlite_out("flow.db", table, dest)


def _check_for_extra_agents(agent: ConversableAgent) -> list[ConversableAgent]:
    _extra_agents: list[ConversableAgent] = []
    _agent_cls_name = agent.__class__.__name__
    if _agent_cls_name == "CaptainAgent":
        _assistant_agent = getattr(agent, "assistant", None)
        if _assistant_agent and _assistant_agent not in _extra_agents:
            _extra_agents.append(_assistant_agent)
        _executor_agent = getattr(agent, "executor", None)
        if _executor_agent and _executor_agent not in _extra_agents:
            _extra_agents.append(_executor_agent)
    return _extra_agents


def _check_for_group_members(agent: ConversableAgent) -> list[ConversableAgent]:
    _extra_agents: list[ConversableAgent] = []
    _group_chat = getattr(agent, "_groupchat", None)
    if _group_chat:
        _chat_agents = getattr(_group_chat, "agents", [])
        if isinstance(_chat_agents, list):
            for _group_member in _chat_agents:
                if _group_member not in _extra_agents:
                    _extra_agents.append(_group_member)
    _manager = getattr(agent, "_group_manager", None)
    if _manager:
        if _manager not in _extra_agents:
            _extra_agents.append(_manager)
        for _group_member in _check_for_group_members(_manager):
            if _group_member not in _extra_agents:
                _extra_agents.append(_group_member)
    return _extra_agents


def _get_known_agents() -> list[ConversableAgent]:
    _known_agents: list[ConversableAgent] = []
    if customer_proxy not in _known_agents:
        _known_agents.append(customer_proxy)
    _known_agents.append(customer_proxy)
    for _group_member in _check_for_group_members(customer_proxy):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(customer_proxy):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if customer_engagement_agent not in _known_agents:
        _known_agents.append(customer_engagement_agent)
    _known_agents.append(customer_engagement_agent)
    for _group_member in _check_for_group_members(customer_engagement_agent):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(customer_engagement_agent):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if topic_preference_agent not in _known_agents:
        _known_agents.append(topic_preference_agent)
    _known_agents.append(topic_preference_agent)
    for _group_member in _check_for_group_members(topic_preference_agent):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(topic_preference_agent):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if personal_information_agent not in _known_agents:
        _known_agents.append(personal_information_agent)
    _known_agents.append(personal_information_agent)
    for _group_member in _check_for_group_members(personal_information_agent):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(personal_information_agent):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)
    return _known_agents


async def store_error(exc: BaseException | None = None) -> None:
    """Store the error in error.json.

    Parameters
    ----------
    exc : BaseException | None
        The exception we got if any.
    """
    reason = "Event handler stopped processing" if not exc else traceback.format_exc()
    try:
        async with aiofiles.open(
            "error.json", "w", encoding="utf-8", newline="\n"
        ) as file:
            await file.write(json.dumps({"error": reason}))
    except BaseException:  # pylint: disable=broad-exception-caught
        pass


async def store_results(result_dicts: list[dict[str, Any]]) -> None:
    """Store the results to results.json.
    Parameters
    ----------
    result_dicts : list[dict[str, Any]]
        The list of the results.
    """
    async with aiofiles.open(
        "results.json", "w", encoding="utf-8", newline="\n"
    ) as file:
        await file.write(
            json.dumps({"results": result_dicts}, indent=4, ensure_ascii=False)
        )


# Start chatting


async def main(
    on_event: (
        Callable[[BaseEvent, list[ConversableAgent]], Coroutine[None, None, bool]]
        | None
    ) = None,
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
    results: list[AsyncRunResponseProtocol] | AsyncRunResponseProtocol = []
    result_dicts: list[dict[str, Any]] = []
    results = await personal_information_agent.a_sequential_run(
        [
            {
                "recipient": customer_proxy,
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "summary_prompt": "Return the customer information into as JSON object only: {'name': '', 'location': ''}",
                    "summary_role": "user",
                },
                "max_turns": 1,
                "clear_history": True,
                "chat_id": 0,
                "message": "Hello, I'm here to help you get started with our product. Could you tell me your name and location?",
            },
            {
                "sender": topic_preference_agent,
                "recipient": customer_proxy,
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "summary_prompt": "Return the customer information into as JSON object only: {'topic_of_interest': ''}",
                    "summary_role": "user",
                },
                "max_turns": 1,
                "clear_history": True,
                "chat_id": 1,
                "prerequisites": [0],
                "message": "Great! Could you tell me what topics you are interested in reading about?",
            },
            {
                "sender": customer_proxy,
                "recipient": customer_engagement_agent,
                "summary_method": "last_msg",
                "max_turns": 1,
                "clear_history": True,
                "chat_id": 2,
                "prerequisites": [1],
                "message": "Let's find something fun to read.",
            },
        ]
    )
    if not isinstance(results, list):
        results = [results]  # pylint: disable=redefined-variable-type
    got_agents = False
    known_agents: list[ConversableAgent] = []
    result_events: list[dict[str, Any]] = []
    if on_event:
        for index, result in enumerate(results):
            result_events = []
            async for event in result.events:
                try:
                    result_events.append(event.model_dump(mode="json", fallback=str))
                except BaseException:  # pylint: disable=broad-exception-caught
                    pass
                if not got_agents:
                    known_agents = _get_known_agents()
                    got_agents = True
                try:
                    should_continue = await on_event(event, known_agents)
                except BaseException as e:
                    await stop_logging()
                    await store_error(e)
                    raise SystemExit("Error in event handler: " + str(e)) from e
                if getattr(event, "type") == "run_completion":
                    break
                if not should_continue:
                    await stop_logging()
                    await store_error()
                    raise SystemExit("Event handler stopped processing")
            result_cost = await result.cost
            result_context_variables = await result.context_variables
            result_dict = {
                "index": index,
                "uuid": str(result.uuid),
                "events": result_events,
                "messages": await result.messages,
                "summary": await result.summary,
                "cost": (
                    result_cost.model_dump(mode="json", fallback=str)
                    if result_cost
                    else None
                ),
                "context_variables": (
                    result_context_variables.model_dump(mode="json", fallback=str)
                    if result_context_variables
                    else None
                ),
                "last_speaker": await result.last_speaker,
            }
            result_dicts.append(result_dict)
    else:
        for index, result in enumerate(results):
            result_events = []
            await result.process()
            async for event in result.events:
                try:
                    result_events.append(event.model_dump(mode="json", fallback=str))
                except BaseException:  # pylint: disable=broad-exception-caught
                    pass
            result_cost = await result.cost
            result_context_variables = await result.context_variables
            result_dict = {
                "index": index,
                "uuid": str(result.uuid),
                "events": result_events,
                "messages": await result.messages,
                "summary": await result.summary,
                "cost": (
                    result_cost.model_dump(mode="json", fallback=str)
                    if result_cost
                    else None
                ),
                "context_variables": (
                    result_context_variables.model_dump(mode="json", fallback=str)
                    if result_context_variables
                    else None
                ),
                "last_speaker": await result.last_speaker,
            }
            result_dicts.append(result_dict)

    await stop_logging()
    await store_results(result_dicts)
    return result_dicts


async def call_main() -> None:
    """Run the main function and print the results."""
    results: list[dict[str, Any]] = await main()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Let's go!
    anyio.run(call_main)
