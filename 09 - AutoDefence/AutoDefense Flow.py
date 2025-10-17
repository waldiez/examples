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

"""AutoDefense Flow.

Multi-Agent LLM Defense against Jailbreak Attacks

Requirements: ag2[openai]==0.9.10
Tags: AutoDefense, Jailbreak
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
# "autodefense_flow_api_keys.py"
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


__MODELS_MODULE__ = load_api_key_module("autodefense_flow")


def get_autodefense_flow_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_autodefense_flow_model_api_key(model_name)


# Models

gpt_3_5_turbo_llm_config: dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "api_type": "openai",
    "api_key": get_autodefense_flow_model_api_key("gpt_3_5_turbo"),
}

# Agents

answer_provider = AssistantAgent(
    name="answer_provider",
    description="Final answer provider",
    system_message="Provide an answer to the original user's question. Consider the content provided by the other agents. In case the question the judge think it is harmful just say: \"I'm sorry, but I cannot fulfill this request.\" otherwise answer the questions using two sentences.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

intention_analyzer = AssistantAgent(
    name="intention_analyzer",
    description="Intention Analyzer",
    system_message="Your role is to analyze the given content and present a comprehensive analysis\nof the intended message behind the text, including its context and any implied meanings.\nYou should present a fair argument without judgment.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

interface = AssistantAgent(
    name="interface",
    description="You are just the input and output of the guardrail system",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

judge = AssistantAgent(
    name="judge",
    description="Judge",
    system_message="Your role is to evaluate the response presented by Intention Analyzer and Original Prompt Analyzer. Consider the intention and prompt inference to deliver a judgment on whether the system input content is valid or invalid.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

original_prompt_analyzer = AssistantAgent(
    name="original_prompt_analyzer",
    description="Original Prompt Analyzer",
    system_message="Your task is to infer the original prompt that led to the given LLM output, you should present three possible prompts that may produce this output. Please use the context provided by Intention Analyzer.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_3_5_turbo_llm_config,
        ],
        cache_seed=42,
    ),
)

user = UserProxyAgent(
    name="user",
    description="A new User proxy agent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    llm_config=False,
)


def nested_chat_message_analyzeintent(
    recipient: ConversableAgent,
    messages: list[dict[str, Any]],
    sender: ConversableAgent,
    config: dict[str, Any],
) -> Union[dict[str, Any], str]:
    """Ask for a review."""
    return f"""Review the following content.
{recipient.chat_messages_for_summary(sender)[-1]['content']}"""


def nested_chat_message_analyzeoriginalprompt(
    recipient: ConversableAgent,
    messages: list[dict[str, Any]],
    sender: ConversableAgent,
    config: dict[str, Any],
) -> Union[dict[str, Any], str]:
    """Ask for a review."""
    return f"""Review the following content.
{recipient.chat_messages_for_summary(sender)[-1]['content']}"""


def nested_chat_message_judgecontent(
    recipient: ConversableAgent,
    messages: list[dict[str, Any]],
    sender: ConversableAgent,
    config: dict[str, Any],
) -> Union[dict[str, Any], str]:
    """Ask for a review."""
    return f"""Review the following content.
{recipient.chat_messages_for_summary(sender)[-1]['content']}"""


interface_chat_queue: list[dict[str, Any]] = [
    {
        "summary_method": "last_msg",
        "max_turns": 1,
        "clear_history": True,
        "chat_id": 0,
        "recipient": intention_analyzer,
        "message": nested_chat_message_analyzeintent,
    },
    {
        "summary_method": "last_msg",
        "max_turns": 1,
        "clear_history": True,
        "chat_id": 1,
        "recipient": original_prompt_analyzer,
        "message": nested_chat_message_analyzeoriginalprompt,
    },
    {
        "summary_method": "last_msg",
        "max_turns": 1,
        "clear_history": True,
        "chat_id": 2,
        "recipient": judge,
        "message": nested_chat_message_judgecontent,
    },
    {
        "summary_method": "last_msg",
        "max_turns": 1,
        "clear_history": True,
        "chat_id": 3,
        "recipient": answer_provider,
        "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
    },
]

interface.register_nested_chats(
    trigger=["user"],
    chat_queue=interface_chat_queue,
    use_async=False,
    ignore_async_in_sync_chat=True,
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
    with open(json_file, "w", encoding="utf-8", newline="\n") as file:
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
    if user not in _known_agents:
        _known_agents.append(user)
    _known_agents.append(user)
    for _group_member in _check_for_group_members(user):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(user):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if interface not in _known_agents:
        _known_agents.append(interface)
    _known_agents.append(interface)
    for _group_member in _check_for_group_members(interface):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(interface):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if intention_analyzer not in _known_agents:
        _known_agents.append(intention_analyzer)
    _known_agents.append(intention_analyzer)
    for _group_member in _check_for_group_members(intention_analyzer):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(intention_analyzer):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if original_prompt_analyzer not in _known_agents:
        _known_agents.append(original_prompt_analyzer)
    _known_agents.append(original_prompt_analyzer)
    for _group_member in _check_for_group_members(original_prompt_analyzer):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(original_prompt_analyzer):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if judge not in _known_agents:
        _known_agents.append(judge)
    _known_agents.append(judge)
    for _group_member in _check_for_group_members(judge):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(judge):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if answer_provider not in _known_agents:
        _known_agents.append(answer_provider)
    _known_agents.append(answer_provider)
    for _group_member in _check_for_group_members(answer_provider):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(answer_provider):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)
    return _known_agents


def store_error(exc: BaseException | None = None) -> None:
    """Store the error in error.json.

    Parameters
    ----------
    exc : BaseException | None
        The exception we got if any.
    """
    reason = "Event handler stopped processing" if not exc else traceback.format_exc()
    try:
        with open("error.json", "w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps({"error": reason}))
    except BaseException:  # pylint: disable=broad-exception-caught
        pass


def store_results(result_dicts: list[dict[str, Any]]) -> None:
    """Store the results to results.json.
    Parameters
    ----------
    result_dicts : list[dict[str, Any]]
        The list of the results.
    """
    with open("results.json", "w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps({"results": result_dicts}, indent=4, ensure_ascii=False))


# Start chatting


def main(
    on_event: Callable[[BaseEvent, list[ConversableAgent]], bool] | None = None,
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
    with Cache.disk(cache_seed=42) as cache:
        results = interface.run(
            user,
            cache=cache,
            summary_method="last_msg",
            max_turns=3,
            clear_history=True,
            message="Hi there, how I can assist you?",
        )
        if not isinstance(results, list):
            results = [results]  # pylint: disable=redefined-variable-type
        got_agents = False
        known_agents: list[ConversableAgent] = []
        result_events: list[dict[str, Any]] = []
        if on_event:
            for index, result in enumerate(results):
                result_events = []
                for event in result.events:
                    try:
                        result_events.append(
                            event.model_dump(mode="json", fallback=str)
                        )
                    except BaseException:  # pylint: disable=broad-exception-caught
                        pass
                    if not got_agents:
                        known_agents = _get_known_agents()
                        got_agents = True
                    try:
                        should_continue = on_event(event, known_agents)
                    except BaseException as e:
                        stop_logging()
                        store_error(e)
                        raise SystemExit("Error in event handler: " + str(e)) from e
                    if getattr(event, "type") == "run_completion":
                        break
                    if not should_continue:
                        stop_logging()
                        store_error()
                        raise SystemExit("Event handler stopped processing")
                result_cost = result.cost
                result_context_variables = result.context_variables
                result_dict = {
                    "index": index,
                    "uuid": str(result.uuid),
                    "events": result_events,
                    "messages": result.messages,
                    "summary": result.summary,
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
                    "last_speaker": result.last_speaker,
                }
                result_dicts.append(result_dict)
        else:
            for index, result in enumerate(results):
                result_events = []
                result.process()
                for event in result.events:
                    try:
                        result_events.append(
                            event.model_dump(mode="json", fallback=str)
                        )
                    except BaseException:  # pylint: disable=broad-exception-caught
                        pass
                result_cost = result.cost
                result_context_variables = result.context_variables
                result_dict = {
                    "index": index,
                    "uuid": str(result.uuid),
                    "events": result_events,
                    "messages": result.messages,
                    "summary": result.summary,
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
                    "last_speaker": result.last_speaker,
                }
                result_dicts.append(result_dict)

        stop_logging()
    store_results(result_dicts)
    return result_dicts


def call_main() -> None:
    """Run the main function and print the results."""
    results: list[dict[str, Any]] = main()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Let's go!
    call_main()
