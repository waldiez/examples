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
from autogen.agentchat.group import AgentTarget, ContextExpression, ContextVariables, ExpressionAvailableCondition, ExpressionContextCondition, OnContextCondition, ReplyResult, RevertToUserTarget
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.targets.transition_target import AgentNameTarget, AgentTarget, RevertToUserTarget
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

def analyze_request(
    request: Annotated[str, "The user request text to analyze"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Analyze a user request to determine routing based on content
    Updates context variables with routing information
    """
    context_variables["question_answered"] = False

    # Update request tracking
    context_variables["routing_started"] = True
    context_variables["request_count"] += 1
    context_variables["current_request"] = request

    # Previous domain becomes part of history
    if context_variables["current_domain"]:
        prev_domain = context_variables["current_domain"]
        context_variables["previous_domains"].append(prev_domain)
        if prev_domain in context_variables["domain_history"]:
            context_variables["domain_history"][prev_domain] += 1
        else:
            context_variables["domain_history"][prev_domain] = 1

    # Reset current_domain to be determined by the router
    context_variables["current_domain"] = None

    return ReplyResult(
        message=f"Request analyzed. Will determine the best specialist to handle: '{request}'",
        context_variables=context_variables
    )

def route_to_tech_specialist(
    confidence: Annotated[int, "Confidence level for tech domain (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to tech specialist"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the technology specialist
    """
    context_variables["current_domain"] = "technology"
    context_variables["domain_confidence"]["technology"] = confidence
    context_variables["tech_invocations"] += 1

    return ReplyResult(
        target=AgentTarget(agent=tech_specialist),
        message=f"Routing to tech specialist with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables
    )

def route_to_finance_specialist(
    confidence: Annotated[int, "Confidence level for finance domain (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to finance specialist"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the finance specialist
    """
    context_variables["current_domain"] = "finance"
    context_variables["domain_confidence"]["finance"] = confidence
    context_variables["finance_invocations"] += 1

    return ReplyResult(
        #target=AgentTarget(finance_specialist),
        target=AgentNameTarget(agent_name="finance_specialist"),
        message=f"Routing to finance specialist with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables)

def route_to_healthcare_specialist(
    confidence: Annotated[int, "Confidence level for healthcare domain (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to healthcare specialist"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the healthcare specialist
    """
    context_variables["current_domain"] = "healthcare"
    context_variables["domain_confidence"]["healthcare"] = confidence
    context_variables["healthcare_invocations"] += 1

    return ReplyResult(
        target=AgentTarget(agent=healthcare_specialist),
        message=f"Routing to healthcare specialist with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables
    )

def route_to_general_specialist(
    confidence: Annotated[int, "Confidence level for general domain (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to general knowledge specialist"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the general knowledge specialist
    """
    context_variables["current_domain"] = "general"
    context_variables["domain_confidence"]["general"] = confidence
    context_variables["general_invocations"] += 1

    return ReplyResult(
        target=AgentTarget(agent=general_specialist),
        message=f"Routing to general knowledge specialist with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables
    )

def provide_tech_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from the technology specialist
    """
    # Record the question and response
    context_variables["question_responses"].append({
        "domain": "technology",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True

    return ReplyResult(
        message="Technology specialist response provided.",
        context_variables=context_variables
    )

def provide_finance_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from the finance specialist
    """
    # Record the question and response
    context_variables["question_responses"].append({
        "domain": "finance",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True

    return ReplyResult(
        message="Finance specialist response provided.",
        context_variables=context_variables
    )

def provide_healthcare_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from the healthcare specialist
    """
    # Record the question and response
    context_variables["question_responses"].append({
        "domain": "healthcare",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True

    return ReplyResult(
        message="Healthcare specialist response provided.",
        context_variables=context_variables
    )

def provide_general_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from the general knowledge specialist
    """
    # Record the question and response
    context_variables["question_responses"].append({
        "domain": "general",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True

    return ReplyResult(
        message="General knowledge specialist response provided.",
        context_variables=context_variables
    )

# Function for follow-up clarification if needed
def request_clarification(
    clarification_question: Annotated[str, "Question to ask user for clarification"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Request clarification from the user when the query is ambiguous
    """
    return ReplyResult(
        message=f"Further clarification is required to determine the correct domain: {clarification_question}",
        context_variables=context_variables,
        target=RevertToUserTarget()
    )

# Models

gpt_4_1_mini_llm_config: dict[str, Any] = {
    "model": "gpt-4.1-mini",
    "api_type": "openai",
    "api_key": get_waldiez_flow_model_api_key("gpt_4_1_mini")
}

# Agents

finance_specialist = ConversableAgent(
    name="finance_specialist",
    description="A new Assistant agent",
    system_message="You are the finance specialist with deep expertise in personal finance, investments, banking, budgeting, financial planning, taxes, economics, and business finance.\n\n    When responding to queries in your domain:\n    1. Provide accurate financial information and advice based on sound financial principles\n    2. Explain financial concepts clearly without excessive jargon\n    3. Present balanced perspectives on financial decisions, acknowledging risks and benefits\n    4. Avoid making specific investment recommendations but provide educational information about investment types\n    5. Include relevant financial principles, terms, or calculations when appropriate\n\n    Focus on being informative, balanced, and helpful. If a query contains elements outside your domain of expertise, focus on the financial aspects while acknowledging the broader context.\n\n    Use the provide_finance_response tool to submit your final response.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        provide_finance_response,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

general_specialist = ConversableAgent(
    name="general_specialist",
    description="A new Assistant agent",
    system_message="You are the general knowledge specialist with broad expertise across multiple domains and topics.\n\n    When responding to queries in your domain:\n    1. Provide comprehensive information drawing from relevant knowledge domains\n    2. Handle questions that span multiple domains or don't clearly fit into a specialized area\n    3. Synthesize information from different fields when appropriate\n    4. Provide balanced perspectives on complex topics\n    5. Address queries about history, culture, society, ethics, environment, education, arts, and other general topics\n\n    Focus on being informative, balanced, and helpful. For questions that might benefit from deeper domain expertise, acknowledge this while providing the best general information possible.\n\n    Use the provide_general_response tool to submit your final response.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        provide_general_response,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

healthcare_specialist = ConversableAgent(
    name="healthcare_specialist",
    description="A new Assistant agent",
    system_message="You are the healthcare specialist with deep expertise in health, medicine, fitness, nutrition, diseases, medical conditions, and wellness.\n\n    When responding to queries in your domain:\n    1. Provide accurate health information based on current medical understanding\n    2. Explain medical concepts in clear, accessible language\n    3. Include preventive advice and best practices for health management when appropriate\n    4. Reference relevant health principles, systems, or processes\n    5. Always clarify that you're providing general information, not personalized medical advice\n\n    Focus on being informative, accurate, and helpful. If a query contains elements outside your domain of expertise, focus on the health aspects while acknowledging the broader context.\n\n    Use the provide_healthcare_response tool to submit your final response.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        provide_healthcare_response,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

router_agent = ConversableAgent(
    name="router_agent",
    description="A new Assistant agent",
    system_message="You are the routing agent responsible for analyzing user requests and directing them to the most appropriate specialist.\n\n    Your task is to carefully analyze each user query and determine which domain specialist would be best equipped to handle it:\n\n    1. Technology Specialist: For questions about computers, software, programming, IT issues, electronics, digital tools, internet, etc. Use route_to_tech_specialist to transfer.\n    2. Finance Specialist: For questions about money, investments, banking, budgeting, financial planning, taxes, economics, etc. Use route_to_finance_specialist to transfer.\n    3. Healthcare Specialist: For questions about health, medicine, fitness, nutrition, diseases, medical conditions, wellness, etc. Use route_to_healthcare_specialist to transfer.\n    4. General Knowledge Specialist: For general questions that don't clearly fit the other categories or span multiple domains. Use route_to_general_specialist to transfer.\n\n    For each query, you must:\n    1. Use the analyze_request tool to process the query and update context\n    2. Determine the correct domain by analyzing keywords, themes, and context\n    3. Consider the conversation history and previous domains if available\n    4. Route to the most appropriate specialist using the corresponding routing tool\n\n    When routing:\n    - Provide a confidence level (1-10) based on how certain you are about the domain\n    - Include detailed reasoning for your routing decision\n    - If a query seems ambiguous or spans multiple domains, route to the specialist who can best handle the primary intent\n\n    Always maintain context awareness by considering:\n    - Current query content and intent\n    - Previously discussed topics\n    - User's possible follow-up patterns\n    - Domain switches that might indicate changing topics\n\n    After a specialist has provided an answer, output the question and answer.\n\n    For ambiguous queries that could belong to multiple domains:\n    - If you are CERTAIN that the query is multi-domain but has a primary focus, route to the specialist for that primary domain\n    - If you are NOT CERTAIN and there is no clear primary domain, use the request_clarification tool to ask the user for more specifics\n    - When a query follows up on a previous topic, consider maintaining consistency by routing to the same specialist unless the domain has clearly changed",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        analyze_request,
        route_to_finance_specialist,
        route_to_healthcare_specialist,
        route_to_tech_specialist,
        route_to_general_specialist,
        request_clarification,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_mini_llm_config,
        ],
        cache_seed=42,
    ),
)

tech_specialist = ConversableAgent(
    name="tech_specialist",
    description="A new Assistant agent",
    system_message="You are the technology specialist with deep expertise in computers, software, programming, IT, electronics, digital tools, and internet technologies.\n\n    When responding to queries in your domain:\n    1. Provide accurate, technical information based on current industry knowledge\n    2. Explain complex concepts in clear terms appropriate for the user's apparent level of technical understanding\n    3. Include practical advice, troubleshooting steps, or implementation guidance when applicable\n    4. Reference relevant technologies, programming languages, frameworks, or tools as appropriate\n    5. For coding questions, provide correct, well-structured code examples when helpful\n\n    Focus on being informative, precise, and helpful. If a query contains elements outside your domain of expertise, focus on the technology aspects while acknowledging the broader context.\n\n    Use the provide_tech_response tool to submit your final response.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        provide_tech_response,
    ],
    update_agent_state_before_reply=[],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_mini_llm_config,
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

finance_specialist.handoffs.set_after_work(
    target=AgentTarget(router_agent)
)

general_specialist.handoffs.set_after_work(
    target=AgentTarget(router_agent)
)

healthcare_specialist.handoffs.set_after_work(
    target=AgentTarget(router_agent)
)

router_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(tech_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("${current_domain} == 'technology'")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("!${question_answered}")
        ),
    )
)
router_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(finance_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("${current_domain} == 'finance'")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("!${question_answered}")
        ),
    )
)
router_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(healthcare_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("${current_domain} == 'healthcare'")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("!${question_answered}")
        ),
    )
)
router_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(general_specialist),
        condition=ExpressionContextCondition(expression=ContextExpression("${current_domain} == 'general'")),
        available=ExpressionAvailableCondition(
            expression=ContextExpression("!${question_answered}")
        ),
    )
)
router_agent.handoffs.set_after_work(
    target=RevertToUserTarget()
)

tech_specialist.handoffs.set_after_work(
    target=AgentTarget(router_agent)
)

manager_pattern = DefaultPattern(
    initial_agent=router_agent,
    agents=[router_agent, tech_specialist, finance_specialist, healthcare_specialist, general_specialist],
    user_agent=user,
    group_manager_args={
        "llm_config": False
    },
    context_variables=ContextVariables(data={
        "routing_started": False,
        "current_domain": None,
        "previous_domains": [],
        "domain_confidence": {},
        "request_count": 0,
        "current_request": "",
        "domain_history": {},
        "question_responses": [],
        "question_answered": True,
        "tech_invocations": 0,
        "finance_invocations": 0,
        "healthcare_invocations": 0,
        "general_invocations": 0,
        "has_error": False,
        "error_message": "",
    }),
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
            messages="I have a question. Can you tell me about benefits? I'm trying to understand all my options and make the right decision.",
            max_rounds=100,
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
