import os

from datapizza.clients.factory import ClientFactory
from datapizza.core.clients import Client

DEFAULT_PROVIDER = "openai"


def create_client(model_env: str = "MODEL_AGENT", system_prompt: str = "", temperature: float = 0.7) -> Client:
    
    model = os.environ.get(model_env, "")
    if not model:
        raise RuntimeError(f"Variable {model_env} isn't set in .env")

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("Variable LLM_API_KEY isn't set in .env")

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    return ClientFactory.create(
        provider=provider,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
    )
