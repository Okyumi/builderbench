import logging
import time
from collections import namedtuple

from google import genai
from google.genai import types

from anthropic import Anthropic
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "auxiliary",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLING_PARAMS = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
        },

    },
}

def resolve_sampling_params(model_id, generate_kwargs=None):
    """Resolve sampling parameters for a given model, with optional overrides."""
    resolved = {}
    resolved.update(DEFAULT_SAMPLING_PARAMS.get(model_id, {}))
    
    if generate_kwargs is not None:
        resolved.update(generate_kwargs)

    return resolved

class LLMClientWrapper:

    def __init__(self, client_config):
        """Store client name, model ID, base URL, and resolved sampling params from config."""
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url

        self.sampling_params = resolve_sampling_params(
            model_id=self.model_id,
            generate_kwargs=client_config.generate_kwargs,
        )

    def generate(self, messages):
        """Generate a response from the LLM; must be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses")

class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAI wrapper with configuration and defer client creation."""
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            logger.info("Initialised OpenAI-compatible client (backend=%s, model=%s)", self.client_name, self.model_id)
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the text-only format expected by the OpenAI API."""
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                raise NotImplementedError("Attachments are not supported in this implementation.")
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Call the OpenAI chat completions API and return a normalized LLMResponse."""
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        # Create kwargs for the API call
        api_kwargs = {
            "model": self.model_id,
            "messages": converted_messages,
        }
        api_kwargs.update(self.sampling_params)

        response = self.client.chat.completions.create(**api_kwargs)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            auxiliary=None,
        )

class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        """Initialize the Claude wrapper with configuration and defer client creation."""
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            self.client = Anthropic()
            logger.info("Initialised Claude client (model=%s)", self.model_id)
            self._initialized = True

    def convert_messages(self, messages):
        """Separate system prompt from user/assistant messages for the Claude API format."""
        converted_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    logger.warning("Warning: Multiple system messages detected. Concatenating their content and using as a single system prompt.")
                    system_prompt += "\n\n" + msg.content
                continue

            if msg.attachment is not None:
                raise NotImplementedError("Attachments are not supported in this implementation.")

            converted_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return system_prompt, converted_messages

    def generate(self, messages):
        """Call the Claude messages API and return a normalized LLMResponse."""
        self._initialize_client()
        system_prompt, converted_messages = self.convert_messages(messages)

        # Create kwargs for the API call
        api_kwargs = {
            "model": self.model_id,
            "messages": converted_messages,
        }
        
        if system_prompt:
            api_kwargs["system"] = system_prompt

        api_kwargs.update(self.sampling_params)
        
        response = self.client.messages.create(**api_kwargs)
        
        text_content = next(
            (block.text for block in response.content if block.type == "text"), 
            ""
        ).strip()

        return LLMResponse(
            model_id=self.model_id,
            completion=text_content,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            auxiliary=None,
        )

class GeminiWrapper(LLMClientWrapper):

    def __init__(self, client_config):
        """Initialize the Gemini wrapper with configuration and defer client creation."""
        super().__init__(client_config)
        self._initialized = False
    
    def _initialize_client(self):
        """Initialize the Gemini client if not already initialized."""
        if not self._initialized:
            # Assumes GEMINI_API_KEY is set in your environment variables
            self.client = genai.Client()
            self._initialized = True

    def convert_messages(self, messages):
        """Separate system prompt and convert messages to Gemini Content objects."""
        converted_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    logger.warning("Warning: Multiple system messages detected. Concatenating their content and using as a single system prompt.")
                    system_prompt += "\n\n" + msg.content
                continue

            if msg.attachment is not None:
                raise NotImplementedError("Attachments are not supported in this implementation.")

            role = "model" if msg.role == "assistant" else msg.role
            converted_messages.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                )
            )

        return system_prompt, converted_messages
    
    def generate(self, messages):
        """Call the Gemini generate_content API and return a normalized LLMResponse."""
        self._initialize_client()
        system_prompt, converted_messages = self.convert_messages(messages)

        config_kwargs = {}
        config_kwargs.update(self.sampling_params)
        if "thinking_config" in config_kwargs and isinstance(config_kwargs["thinking_config"], dict):
            config_kwargs["thinking_config"] = types.ThinkingConfig(**config_kwargs["thinking_config"])
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        config = types.GenerateContentConfig(**config_kwargs)

        # Execute generation
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=converted_messages,
            config=config
        )

        completion_text = ""
        thoughts_text = ""
        stop_reason = None
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.finish_reason:
                stop_reason = candidate.finish_reason.name
            
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if not part.text:
                        continue

                    if getattr(part, 'thought', False): 
                        thoughts_text += part.text + "\n"
                    else:
                        completion_text += part.text + "\n"
        
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        auxiliary_data = f"Internal thought summary:\n {thoughts_text.strip()}" if thoughts_text else None

        return LLMResponse(
            model_id=self.model_id,
            completion=completion_text.strip() if completion_text else "",
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            auxiliary=auxiliary_data,
        )

def create_llm_client(client_config):
    """Return a factory function that instantiates the correct LLM wrapper based on client name."""
    def client_factory():
        """Instantiate and return the appropriate LLM wrapper for the configured client."""
        client_name_lower = client_config.client_name.lower()
        if "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            return OpenAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GeminiWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory