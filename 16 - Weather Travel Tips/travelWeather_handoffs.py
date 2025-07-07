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

"""Weather sightseeing recommendation.

A group chat workflow checking whether the weather conditions are fine for visiting a specified site at a specified date. It contains an agent using tool to retrieve the temperature at real-time. The communication within the agents is achieved using handoffs.

Requirements: ag2[openai]==0.9.5
Tags: Weather, Travel, Group
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
from datetime import timedelta
from pprint import pprint
from types import ModuleType
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import autogen  # type: ignore
from autogen import (
    Agent,
    Cache,
    ChatResult,
    ConversableAgent,
    GroupChat,
    UpdateSystemMessage,
    UserProxyAgent,
    register_function,
    runtime_logging,
)
from autogen.agentchat import GroupChatManager, initiate_group_chat
from autogen.agentchat.group import (
    AgentTarget,
    ContextVariables,
    OnCondition,
    OnContextCondition,
    ReplyResult,
    RevertToUserTarget,
    StringContextCondition,
    StringLLMCondition,
)
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.coding import LocalCommandLineCodeExecutor
import numpy as np
import pandas as pd
import requests

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
# This section assumes that a file named "weather_sightseeing_api_keys"
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


__MODELS_MODULE__ = load_api_key_module("weather_sightseeing")


def get_weather_sightseeing_model_api_key(model_name: str) -> str:
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
    return __MODELS_MODULE__.get_weather_sightseeing_model_api_key(model_name)


# Tools


def record_info(
    date: str, time: str, city: str, context_variables: ContextVariables
) -> ReplyResult:
    """Record the date, time and city in the workflow context"""

    context_variables["date"] = date
    context_variables["definedDate"] = True
    context_variables["time"] = time
    context_variables["definedTime"] = True
    context_variables["city"] = city
    context_variables["definedCity"] = True
    context_variables["retrievedInfo"] = True

    return ReplyResult(
        context_variables=context_variables,
        message=f"Info Recorded: {date}, {time} and {city}",
    )


def record_temperature(context_variables: ContextVariables) -> ReplyResult:
    """Record the place in the workflow context"""

    place = context_variables["city"]
    target_time = context_variables["time"]
    target_date_str = context_variables["date"]

    try:
        # Use pandas to parse date and time flexibly
        datetime_str = f"{target_date_str} {target_time}"
        dt = pd.to_datetime(
            datetime_str, dayfirst=True
        )  # dayfirst=True handles DD/MM/YYYY
        hour = dt.hour
        remainder = hour % 3
        if remainder < 1.5:
            rounded_hour = hour - remainder
        else:
            rounded_hour = hour + (3 - remainder)
            if rounded_hour >= 24:
                dt += timedelta(days=1)
                rounded_hour = 0
        dt = dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)

        # Format inputs for API
        place = place.strip()
        formatted_date = dt.strftime("%Y-%m-%d")
        formatted_time = str(dt.hour * 100)

        print(
            f"Searching for weather in {place} on {formatted_date} at {dt.hour:02d}:00..."
        )

        # Get weather data
        response = requests.get(f"https://wttr.in/{place}?format=j1")
        response.raise_for_status()
        data = response.json()

        # Search for the target date and time
        forecast = None
        for day in data["weather"]:
            if day["date"] == formatted_date:
                for slot in day["hourly"]:
                    if slot["time"] == formatted_time:
                        forecast = slot
                        break
                break

        # Output result
        if forecast:
            temp_c = forecast["tempC"]
            feels_like = forecast["FeelsLikeC"]
            desc = forecast["weatherDesc"][0]["value"]
            print(f"\nWeather in {place} on {formatted_date} at {dt.hour:02d}:00:")
            print(f"Temperature: {temp_c}°C, Feels like: {feels_like}°C")
            print(f"Conditions: {desc}")
            context_variables["definedTemperature"] = True
            context_variables["temperature"] = temp_c
        else:
            print(
                f"\nSorry, could not find the forecast for {place} on {formatted_date} at {dt.hour:02d}:00."
            )

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Try formats like: '27/06/2025 2PM', '2025-06-27 14:00', 'June 27, 2025 2:30 PM'"
        )

    return ReplyResult(
        context_variables=context_variables, message=f"Temperature Recorded: {temp_c}"
    )


# Models

gpt_4_1_llm_config: dict[str, Any] = {
    "model": "gpt-4.1",
    "api_type": "openai",
    "api_key": get_weather_sightseeing_model_api_key("gpt_4_1"),
}

# Agents

info_agent_executor = LocalCommandLineCodeExecutor(
    work_dir="coding",
)

info_agent = ConversableAgent(
    name="info_agent",
    description="A place agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config={"executor": info_agent_executor},
    is_termination_msg=None,  # pyright: ignore
    functions=[
        record_info,
    ],
    update_agent_state_before_reply=[
        UpdateSystemMessage("You need to retrieve the city the date and the time"),
    ],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=None,
    ),
)

triage_agent = ConversableAgent(
    name="triage_agent",
    description="triage_agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[],
    update_agent_state_before_reply=[
        UpdateSystemMessage(
            "You are an order triage agent, working with a user and a group of agents to provide support for your weather tips.\n Give the speech to Info_Agent if the user hasn't defined a place.\nThe Weather_Agent will retrieve all weather related tasks. \nYou will manage all weather optimization task related tasks. Be sure to the temperature value first. Then if it's valid you can record it in the context.\n\nAsk the user for further information when necessary.\n\nThe current status of this workflow is:\nCity of interest: {city}\nCity defined: {definedCity}\nTime: {time}\nTime defined: {definedTime}\nDate: {date}\nTemperature: {temperature}"
        ),
    ],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=None,
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

weather_agent = ConversableAgent(
    name="weather_agent",
    description="weather_agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=None,
    default_auto_reply="",
    code_execution_config=False,
    is_termination_msg=None,  # pyright: ignore
    functions=[
        record_temperature,
    ],
    update_agent_state_before_reply=[
        UpdateSystemMessage(
            "You are a weather agent, get temperature data. Check weather the temperature values are safe for the user.\nReturn to the triage_agent if temp is retrieved. \n"
        ),
    ],
    llm_config=autogen.LLMConfig(
        config_list=[
            gpt_4_1_llm_config,
        ],
        cache_seed=None,
    ),
)

info_agent.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(triage_agent),
        condition=StringLLMCondition(prompt="The info have been retrieved"),
    )
)
info_agent.handoffs.set_after_work(target=RevertToUserTarget())

triage_agent.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(info_agent),
        condition=StringLLMCondition(
            prompt="The user hasn't defined a city, ask for the place of interest"
        ),
    )
)
triage_agent.handoffs.add_llm_condition(
    condition=OnCondition(
        target=AgentTarget(weather_agent),
        condition=StringLLMCondition(prompt="Temperature has not been retrieved"),
    )
)
triage_agent.handoffs.set_after_work(target=RevertToUserTarget())

weather_agent.handoffs.add_context_condition(
    condition=OnContextCondition(
        target=AgentTarget(triage_agent),
        condition=StringContextCondition(variable_name="{definedTemperature}"),
    )
)
weather_agent.handoffs.set_after_work(target=AgentTarget(triage_agent))

manager_pattern = DefaultPattern(
    initial_agent=triage_agent,
    agents=[weather_agent, info_agent, triage_agent],
    user_agent=user,
    group_manager_args={"llm_config": False},
    context_variables=ContextVariables(
        data={
            "timestamp": None,
            "temperature": None,
            "city": None,
            "definedCity": False,
            "date": None,
            "definedTime": False,
            "definedDate": False,
            "time": None,
            "retrievedInfo": False,
            "definedTemperature": False,
        }
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


def main() -> Union[ChatResult, list[ChatResult], dict[int, ChatResult]]:
    """Start chatting.

    Returns
    -------
    Union[ChatResult, list[ChatResult], dict[int, ChatResult]]
        The result of the chat session, which can be a single ChatResult,
        a list of ChatResults, or a dictionary mapping integers to ChatResults.
    """
    results, _, __ = initiate_group_chat(
        pattern=manager_pattern,
        messages="Hi I want to visit a place",
        max_rounds=40,
    )

    stop_logging()
    return results


def call_main() -> None:
    """Run the main function and print the results."""
    results: Union[ChatResult, list[ChatResult], dict[int, ChatResult]] = main()
    if isinstance(results, dict):
        # order by key
        ordered_results = dict(sorted(results.items()))
        for _, result in ordered_results.items():
            pprint(asdict(result))
    else:
        if not isinstance(results, list):
            results = [results]
        for result in results:
            pprint(asdict(result))


if __name__ == "__main__":
    # Let's go!
    call_main()
