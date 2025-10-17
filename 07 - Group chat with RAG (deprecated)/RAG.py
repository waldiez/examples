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

"""RAG.

Group Chat with Retrieval Augmented Generation.

Requirements: ag2[openai]==0.9.10, beautifulsoup4, chromadb>=0.5.23, ipython, markdownify, protobuf==5.29.3, pypdf, sentence_transformers
Tags: RAG, FLAML
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
    Cache,
    ChatResult,
    ConversableAgent,
    GroupChat,
    runtime_logging,
)
from autogen.agentchat import GroupChatManager, run_group_chat
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen.agentchat.group import ContextVariables
from autogen.events import BaseEvent
from autogen.io.run_response import AsyncRunResponseProtocol, RunResponseProtocol
import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
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
# "rag_api_keys.py"
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


__MODELS_MODULE__ = load_api_key_module("rag")


def get_rag_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_rag_model_api_key(model_name)


# Models

gpt_4_1_llm_config: dict[str, Any] = {
    "model": "gpt-4.1",
    "api_type": "openai",
    "api_key": get_rag_model_api_key("gpt_4_1"),
}

# Agents

boss_assistant_client = chromadb.Client(Settings(anonymized_telemetry=False))
boss_assistant_embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
)
boss_assistant_client.get_or_create_collection(
    "autogen-docs",
    embedding_function=boss_assistant_embedding_function,
)

boss_assistant = RetrieveUserProxyAgent(
    name="boss_assistant",
    description="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=3,
    default_auto_reply="Reply 'TERMINATE' if the task is done.",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "").endswith(keyword)
        for keyword in ["TERMINATE"]
    ),
    retrieve_config={
        "task": "default",
        "model": "all-MiniLM-L6-v2",
        "docs_path": [
            r"https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md"
        ],
        "new_docs": True,
        "update_context": True,
        "get_or_create": True,
        "overwrite": False,
        "recursive": True,
        "chunk_mode": "multi_lines",
        "must_break_at_empty_line": True,
        "collection_name": "autogen-docs",
        "distance_threshold": -1.0,
        "vector_db": ChromaVectorDB(
            client=boss_assistant_client,
            embedding_function=boss_assistant_embedding_function,
        ),
        "client": boss_assistant_client,
    },
    llm_config=False,
)

code_reviewer = ConversableAgent(
    name="code_reviewer",
    description="Code Reviewer who can review the code.",
    system_message="You are a code reviewer. Reply 'TERMINATE' in the end when everything is done.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "").endswith(keyword)
        for keyword in ["TERMINATE"]
    ),
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=42,
    ),
)

product_manager = ConversableAgent(
    name="product_manager",
    description="Product Manager who can design and plan the project.",
    system_message="You are a product manager. Reply 'TERMINATE' in the end when everything is done.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "").endswith(keyword)
        for keyword in ["TERMINATE"]
    ),
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=42,
    ),
)

senior_python_engineer = ConversableAgent(
    name="senior_python_engineer",
    description="Senior Python Engineer",
    system_message="You are a senior python engineer, you provide python code to answer questions. Reply 'TERMINATE' in the end when everything is done.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        isinstance(x, dict)
        and x.get("content", "")
        and isinstance(x.get("content", ""), str)
        and x.get("content", "").endswith(keyword)
        for keyword in ["TERMINATE"]
    ),
    functions=[],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=42,
    ),
)

manager_group_chat = GroupChat(
    agents=[product_manager, senior_python_engineer, code_reviewer, boss_assistant],
    enable_clear_history=False,
    send_introductions=False,
    messages=[],
    max_round=20,
    admin_name="boss_assistant",
    speaker_selection_method="round_robin",
    allow_repeat_speaker=True,
)


def callable_message_boss_assistant_to_manager(
    sender: RetrieveUserProxyAgent,
    recipient: ConversableAgent,
    context: dict[str, Any],
) -> Union[dict[str, Any], str]:
    """Get the message using the RAG message generator method.

    Parameters
    ----------
    sender : RetrieveUserProxyAgent
        The source agent.
    recipient : ConversableAgent
        The target agent.
    context : dict[str, Any]
        The context.

    Returns
    -------
    Union[dict[str, Any], str]
        The message to send using the last carryover.
    """
    carryover = context.get("carryover", "")
    if isinstance(carryover, list):
        carryover = carryover[-1]
    if not isinstance(carryover, str):
        if isinstance(carryover, list):
            carryover = carryover[-1]
        elif isinstance(carryover, dict):
            carryover = carryover.get("content", "")
    if not isinstance(carryover, str):
        carryover = ""
    message = sender.message_generator(sender, recipient, context)
    if carryover:
        message += carryover
    return message


manager = GroupChatManager(
    name="manager",
    description="The group manager agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,
    groupchat=manager_group_chat,
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=42,
    ),
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
    if senior_python_engineer not in _known_agents:
        _known_agents.append(senior_python_engineer)
    _known_agents.append(senior_python_engineer)
    for _group_member in _check_for_group_members(senior_python_engineer):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(senior_python_engineer):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if product_manager not in _known_agents:
        _known_agents.append(product_manager)
    _known_agents.append(product_manager)
    for _group_member in _check_for_group_members(product_manager):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(product_manager):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if code_reviewer not in _known_agents:
        _known_agents.append(code_reviewer)
    _known_agents.append(code_reviewer)
    for _group_member in _check_for_group_members(code_reviewer):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(code_reviewer):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if boss_assistant not in _known_agents:
        _known_agents.append(boss_assistant)
    _known_agents.append(boss_assistant)
    for _group_member in _check_for_group_members(boss_assistant):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(boss_assistant):
        if _extra_agent not in _known_agents:
            _known_agents.append(_extra_agent)

    if manager not in _known_agents:
        _known_agents.append(manager)
    _known_agents.append(manager)
    for _group_member in _check_for_group_members(manager):
        if _group_member not in _known_agents:
            _known_agents.append(_group_member)
    for _extra_agent in _check_for_extra_agents(manager):
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
        results = boss_assistant.run(
            manager,
            cache=cache,
            summary_method="last_msg",
            clear_history=True,
            problem="How to use spark for parallel training in FLAML? Give me sample code.",
            message=callable_message_boss_assistant_to_manager,
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
