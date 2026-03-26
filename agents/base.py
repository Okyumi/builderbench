def resolve_agent_config(config):
    """Resolve and instantiate the agent-specific configuration dataclass.

    Args:
        config: Top-level run configuration whose agent_config attribute
            has agent_name (str) and agent_kwargs (dict | None).

    Returns:
        A populated AgentConfig instance for the requested agent.

    Raises:
        ValueError: If agent_name does not match a known agent module.
    """
    name = config.agent_config.agent_name
    if name == "naive":
        from agents.naive import AgentConfig
    elif name == "cot":
        from agents.cot import AgentConfig
    elif name == "reflexion":
        from agents.reflexion import AgentConfig
    else:
        raise ValueError(f"Cannot import agent config for: {name}")
        
    agent_config = AgentConfig()

    if isinstance(config.agent_config.agent_kwargs, dict):
        for key, value in config.agent_config.agent_kwargs.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)

    return agent_config

class BaseAgent:
    """Base class for agents."""

    def __init__(self, client_factory):
        """Initialize the agent with a client."""
        self.client = client_factory()

    def act(self, obs, info):
        """Generate an action based on the observation."""
        raise NotImplementedError

    def reset(self, terminal_obs=None, terminal_info=None):
        """Reset the agent's internal state."""
        raise NotImplementedError