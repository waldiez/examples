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

"""Waldiez Flow.

A waldiez flow

Requirements: ag2[openai]==0.9.8.post1
Tags: 
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
from typing import Annotated, Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

import autogen  # type: ignore
from autogen import Agent, Cache, ChatResult, ConversableAgent, GroupChat, UserProxyAgent, register_function, runtime_logging
from autogen.agentchat import GroupChatManager, run_group_chat
from autogen.agentchat.group import AgentNameTarget, AgentTarget, ContextExpression, ContextVariables, ExpressionAvailableCondition, ExpressionContextCondition, OnCondition, OnContextCondition, ReplyResult, RevertToUserTarget, StringLLMCondition, TerminateTarget
from autogen.agentchat.group.patterns import DefaultPattern
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
# "waldiez_flow_api_keys.py"
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

__MODELS_MODULE__ = load_api_key_module("waldiez_flow")


def get_waldiez_flow_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_waldiez_flow_model_api_key(model_name)


# Tools

def complete_solar_research(research_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Submit solar energy research findings"""
    context_variables["solar_research"] = research_content
    context_variables["specialist_a1_completed"] = True

    # Check if both specialists under Manager A have completed their tasks
    if context_variables["specialist_a1_completed"] and context_variables["specialist_a2_completed"]:
        context_variables["manager_a_completed"] = True

    return ReplyResult(
        message="Solar research completed and stored.",
        context_variables=context_variables,
        target=AgentTarget(renewable_manager),
    )

def complete_wind_research(research_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Submit wind energy research findings"""
    context_variables["wind_research"] = research_content
    context_variables["specialist_a2_completed"] = True

    # Check if both specialists under Manager A have completed their tasks
    if context_variables["specialist_a1_completed"] and context_variables["specialist_a2_completed"]:
        context_variables["manager_a_completed"] = True

    return ReplyResult(
        message="Wind research completed and stored.",
        context_variables=context_variables,
        target=AgentTarget(renewable_manager),
    )

def complete_hydro_research(research_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Submit hydroelectric energy research findings"""
    context_variables["hydro_research"] = research_content
    context_variables["specialist_b1_completed"] = True

    # Check if both specialists under Manager B have completed their tasks
    if context_variables["specialist_b1_completed"] and context_variables["specialist_b2_completed"]:
        context_variables["manager_b_completed"] = True

    return ReplyResult(
        message="Hydroelectric research completed and stored.",
        context_variables=context_variables,
        target=AgentTarget(storage_manager),
    )

def complete_geothermal_research(research_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Submit geothermal energy research findings"""
    context_variables["geothermal_research"] = research_content
    context_variables["specialist_b2_completed"] = True

    # Check if both specialists under Manager B have completed their tasks
    if context_variables["specialist_b1_completed"] and context_variables["specialist_b2_completed"]:
        context_variables["manager_b_completed"] = True

    return ReplyResult(
        message="Geothermal research completed and stored.",
        context_variables=context_variables,
        target=AgentTarget(storage_manager),
    )

def complete_biofuel_research(research_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Submit biofuel research findings"""
    context_variables["biofuel_research"] = research_content
    context_variables["specialist_c1_completed"] = True
    context_variables["manager_c_completed"] = True

    return ReplyResult(
        message="Biofuel research completed and stored.",
        context_variables=context_variables,
        target=AgentTarget(alternative_manager),
    )

def compile_renewable_section(section_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Compile the renewable energy section (solar and wind) for the final report"""
    context_variables["report_sections"]["renewable"] = section_content

    # Check if all managers have submitted their sections
    if all(key in context_variables["report_sections"] for key in ["renewable", "storage", "alternative"]):
        context_variables["executive_review_ready"] = True
        return ReplyResult(
            message="Renewable energy section compiled. All sections are now ready for executive review.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )
    else:
        return ReplyResult(
            message="Renewable energy section compiled and stored.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )

def compile_storage_section(section_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Compile the energy storage section (hydro and geothermal) for the final report"""
    context_variables["report_sections"]["storage"] = section_content

    # Check if all managers have submitted their sections
    if all(key in context_variables["report_sections"] for key in ["renewable", "storage", "alternative"]):
        context_variables["executive_review_ready"] = True
        return ReplyResult(
            message="Energy storage section compiled. All sections are now ready for executive review.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )
    else:
        return ReplyResult(
            message="Energy storage section compiled and stored.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )

def compile_alternative_section(section_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Compile the alternative energy section (biofuels) for the final report"""
    context_variables["report_sections"]["alternative"] = section_content

    # Check if all managers have submitted their sections
    if all(key in context_variables["report_sections"] for key in ["renewable", "storage", "alternative"]):
        context_variables["executive_review_ready"] = True
        return ReplyResult(
            message="Alternative energy section compiled. All sections are now ready for executive review.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )
    else:
        return ReplyResult(
            message="Alternative energy section compiled and stored.",
            context_variables=context_variables,
            target=AgentTarget(executive_agent),
        )

def initiate_research(context_variables: ContextVariables) -> ReplyResult:
    """Initiate the research process by delegating to managers"""
    context_variables["task_started"] = True

    return ReplyResult(
        message="Research initiated. Tasks have been delegated to the renewable energy manager, storage manager, and alternative energy manager.",
        context_variables=context_variables
    )

def compile_final_report(report_content: str, context_variables: ContextVariables) -> ReplyResult:
    """Compile the final comprehensive report from all sections"""
    context_variables["final_report"] = report_content
    context_variables["task_completed"] = True

    return ReplyResult(
        message="Final report compiled successfully. The comprehensive renewable energy report is now complete.",
        context_variables=context_variables,
        target=AgentTarget(user)  # Return to user with final report
    )

# Models

gpt_4o_mini_llm_config: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "api_type": "openai",
    "api_key": get_waldiez_flow_model_api_key("gpt_4o_mini")
}

# Agents

alternative_manager = ConversableAgent(
    name="alternative_manager",
    description="A new Assistant agent",
    system_message="You are the manager for alternative energy solutions, overseeing biofuel research.\n        Your responsibilities include:\n        1. Reviewing the research from your specialist\n        2. Ensuring the information is accurate and comprehensive\n        3. Synthesizing the information into a cohesive section on alternative energy solutions\n        4. Submitting the compiled research to the executive for final report creation\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        compile_alternative_section,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

biofuel_specialist = ConversableAgent(
    name="biofuel_specialist",
    description="A new Assistant agent",
    system_message="You are a specialist in biofuel technologies.\n        Your task is to research and provide concise information about:\n        1. Current state of biofuel technology\n        2. Types of biofuels and their applications\n        3. Cost comparison with fossil fuels\n        4. Major companies and countries leading in biofuel production\n\n        Be thorough but concise. Your research will be used as part of a larger report.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        complete_biofuel_research,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

executive_agent = ConversableAgent(
    name="executive_agent",
    description="A new Assistant agent",
    system_message="You are the executive overseeing the creation of a comprehensive report on renewable energy technologies.\n\n        You have exactly three manager agents reporting to you, each responsible for specific technology domains:\n        1. Renewable Manager - Oversees solar and wind energy research\n        2. Storage Manager - Oversees hydroelectric and geothermal energy research\n        3. Alternative Manager - Oversees biofuel research\n\n        Your responsibilities include:\n        1. Delegating research tasks to these three specific manager agents\n        2. Providing overall direction and ensuring alignment with the project goals\n        3. Reviewing the compiled sections from each manager\n        4. Synthesizing all sections into a cohesive final report with executive summary\n        5. Ensuring the report is comprehensive, balanced, and meets high-quality standards\n\n        Do not create or attempt to delegate to managers that don't exist in this structure.\n\n        The final report should include:\n        - Executive Summary\n        - Introduction to Renewable Energy\n        - Three main sections:\n        * Solar and Wind Energy (from Renewable Manager)\n        * Hydroelectric and Geothermal Energy (from Storage Manager)\n        * Biofuel Technologies (from Alternative Manager)\n        - Comparison of technologies\n        - Future outlook and recommendations",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        initiate_research,
        compile_final_report,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

geothermal_specialist = ConversableAgent(
    name="geothermal_specialist",
    description="A new Assistant agent",
    system_message="You are a specialist in geothermal energy technologies.\n        Your task is to research and provide concise information about:\n        1. Current state of geothermal technology\n        2. Types of geothermal systems and efficiency rates\n        3. Cost comparison with fossil fuels\n        4. Major companies and countries leading in geothermal energy\n\n        Be thorough but concise. Your research will be used as part of a larger report.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        complete_geothermal_research,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

hydro_specialist = ConversableAgent(
    name="hydro_specialist",
    description="A new Assistant agent",
    system_message="You are a specialist in hydroelectric energy technologies.\n        Your task is to research and provide concise information about:\n        1. Current state of hydroelectric technology\n        2. Types of hydroelectric generation (dams, run-of-river, pumped storage)\n        3. Cost comparison with fossil fuels\n        4. Major companies and countries leading in hydroelectric energy\n\n        Be thorough but concise. Your research will be used as part of a larger report.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        complete_hydro_research,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

renewable_manager = ConversableAgent(
    name="renewable_manager",
    description="A new Assistant agent",
    system_message="You are the manager for renewable energy research, specifically overseeing solar and wind energy specialists.\n        Your responsibilities include:\n        1. Reviewing the research from your specialists\n        2. Ensuring the information is accurate and comprehensive\n        3. Synthesizing the information into a cohesive section on renewable energy\n        4. Submitting the compiled research to the executive for final report creation\n\n        You should wait until both specialists have completed their research before compiling your section.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        compile_renewable_section,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

solar_specialist = ConversableAgent(
    name="solar_specialist",
    description="A new Assistant agent",
    system_message="You are a specialist in solar energy technologies.\n        Your task is to research and provide concise information about:\n        1. Current state of solar technology\n        2. Efficiency rates of different types of solar panels\n        3. Cost comparison with fossil fuels\n        4. Major companies and countries leading in solar energy\n\n        Be thorough but concise. Your research will be used as part of a larger report.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        complete_solar_research,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

storage_manager = ConversableAgent(
    name="storage_manager",
    description="A new Assistant agent",
    system_message="You are the manager for energy storage and hydroelectric technologies, overseeing hydroelectric and geothermal energy specialists.\n        Your responsibilities include:\n        1. Reviewing the research from your specialists\n        2. Ensuring the information is accurate and comprehensive\n        3. Synthesizing the information into a cohesive section on energy storage and hydroelectric solutions\n        4. Submitting the compiled research to the executive for final report creation\n\n        You should wait until both specialists have completed their research before compiling your section.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        compile_storage_section,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
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

wind_specialist = ConversableAgent(
    name="wind_specialist",
    description="A new Assistant agent",
    system_message="You are a specialist in wind energy technologies.\n        Your task is to research and provide concise information about:\n        1. Current state of wind technology (onshore/offshore)\n        2. Efficiency rates of modern wind turbines\n        3. Cost comparison with fossil fuels\n        4. Major companies and countries leading in wind energy\n\n        Be thorough but concise. Your research will be used as part of a larger report.\n\n        Use your tools only one at a time.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        complete_wind_research,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4o_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

alternative_manager.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(biofuel_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${specialist_c1_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
alternative_manager.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(executive_agent),
        condition=StringLLMCondition(prompt="Return to the executive with the compiled alternative energy section"),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${manager_c_completed} == True")
        ),
    )
)

biofuel_specialist.handoffs.set_after_work(
    target=AgentTarget(alternative_manager)
)

executive_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(renewable_manager),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${manager_a_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
executive_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(storage_manager),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${manager_b_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
executive_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(alternative_manager),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${manager_c_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
executive_agent.handoffs.set_after_work(
    target=RevertToUserTarget()
)

geothermal_specialist.handoffs.set_after_work(
    target=AgentTarget(renewable_manager)
)

hydro_specialist.handoffs.set_after_work(
    target=AgentTarget(storage_manager)
)

renewable_manager.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(solar_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${specialist_a1_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
renewable_manager.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(wind_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${specialist_a2_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
renewable_manager.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(executive_agent),
        condition=StringLLMCondition(prompt="Return to the executive after your report has been compiled."),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${manager_a_completed} == True")
        ),
    )
)
renewable_manager.handoffs.set_after_work(
    target=AgentTarget(executive_agent)
)

solar_specialist.handoffs.set_after_work(
    target=AgentTarget(renewable_manager)
)

storage_manager.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(hydro_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${specialist_b1_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
storage_manager.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(geothermal_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("not(${specialist_b2_completed})")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${task_started} == True")
        ),
    )
)
storage_manager.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(executive_agent),
        condition=StringLLMCondition(prompt="Return to the executive after your report has been compiled."),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("${manager_b_completed} == True")
        ),
    )
)
storage_manager.handoffs.set_after_work(
    target=AgentTarget(executive_agent)
)

wind_specialist.handoffs.set_after_work(
    target=AgentTarget(renewable_manager)
)

manager_pattern = DefaultPattern(
    initial_agent=executive_agent,
    agents=[solar_specialist, wind_specialist, hydro_specialist, geothermal_specialist, biofuel_specialist, renewable_manager, storage_manager, alternative_manager, executive_agent],
    user_agent=user,
    group_manager_args={
        "llm_config": False
    },
    context_variables=ContextVariables(data={
        "task_started": False,
        "task_completed": False,
        "executive_review_ready": False,
        "manager_a_completed": False,
        "manager_b_completed": False,
        "manager_c_completed": False,
        "specialist_a1_completed": False,
        "specialist_a2_completed": False,
        "specialist_b1_completed": False,
        "specialist_c1_completed": False,
        "specialist_b2_completed": False,
        "solar_research": "",
        "wind_research": "",
        "hydro_research": "",
        "geothermal_research": "",
        "biofuel_research": "",
        "report_sections": {},
        "final_report": "",
    }),
    group_after_work=TerminateTarget(),
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

def main(on_event: Optional[Callable[[BaseEvent], bool]] = None) -> list[dict[str, Any]]:
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
            messages="We need a comprehensive report on the current state of renewable energy technologies. Please coordinate the research and compilation of this report.",
            max_rounds=50,
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
                        raise SystemExit(
                            "Error in event handler: " + str(e)
                        ) from e
                    if event.type == "run_completion":
                        break
                    if not should_continue:
                        raise SystemExit("Event handler stopped processing")
                result_dict = {
                    "index": index,
                    "messages": result.messages,
                    "summary": result.summary,
                    "cost": result.cost.model_dump(mode="json", fallback=str) if result.cost else None,
                    "context_variables": result.context_variables.model_dump(mode="json", fallback=str) if result.context_variables else None,
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
                    "cost": result.cost.model_dump(mode="json", fallback=str) if result.cost else None,
                    "context_variables": result.context_variables.model_dump(mode="json", fallback=str) if result.context_variables else None,
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
