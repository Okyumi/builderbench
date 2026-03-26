from typing import Any

from agents.client import create_llm_client

from .base import BaseAgent
from .cot import CoTAgent
from .naive import NaiveAgent
from .reflexion import ReflexionAgent

def create_agent(config, env_system_prompt=None):
    """Instantiate an agent according to config."""
    client_factory = create_llm_client(config.llm_config)

    name = config.agent_config.agent_name

    if name == "naive":
        return NaiveAgent(client_factory, config, env_system_prompt)
    elif name == "cot":
        return CoTAgent(client_factory, config, env_system_prompt)
    elif name == "reflexion":
        return ReflexionAgent(client_factory, config, env_system_prompt)
    else:
        raise ValueError(f"Unknown agent type: {name}")