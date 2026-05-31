"""
Unified chat-LLM factory.

Set LLM_PROVIDER in .env to choose the backend:
    - deepseek  (default, OpenAI-compatible API)
    - openai
    - anthropic (Claude — uses the native Messages API)
    - gemini    (Google Gemini — uses the native Gen AI API)
    - qwen      (Alibaba DashScope, OpenAI-compatible endpoint)
    - zhipu     (智谱 GLM, OpenAI-compatible endpoint)
    - doubao    (字节豆包 / 火山方舟, OpenAI-compatible endpoint)

Most backends reuse langchain-openai's ChatOpenAI (Chinese vendors all
ship an OpenAI-compatible endpoint). Anthropic uses langchain-anthropic
because Claude has its own Messages API shape.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


# Provider catalogue.
#   flavour = "openai_compat" uses langchain_openai.ChatOpenAI
#   flavour = "anthropic"    uses langchain_anthropic.ChatAnthropic
_PROVIDERS = {
    "deepseek": {
        "flavour": "openai_compat",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },
    "openai": {
        "flavour": "openai_compat",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "base_url": None,
    },
    "anthropic": {
        "flavour": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        # User should override via LLM_MODEL to the exact snapshot they
        # have access to (e.g. claude-opus-4-7-20260401,
        # claude-sonnet-4-5, claude-3-5-sonnet-20241022 ...).
        "default_model": "claude-sonnet-4-5",
        "base_url": None,
    },
    "gemini": {
        "flavour": "gemini",
        "env_key": "GOOGLE_API_KEY",
        # Free tier friendly default. Override via LLM_MODEL=gemini-2.5-pro
        # for deeper reasoning (slower, costs more).
        "default_model": "gemini-2.5-flash",
        "base_url": None,
    },
    "qwen": {
        "flavour": "openai_compat",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "zhipu": {
        "flavour": "openai_compat",
        "env_key": "ZHIPU_API_KEY",
        "default_model": "glm-4-plus",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "doubao": {
        "flavour": "openai_compat",
        "env_key": "ARK_API_KEY",
        "default_model": "doubao-pro-32k",  # user can override to endpoint id
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
}


def get_chat_llm(
    temperature: float = 0.7,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_retries: int = 3,
):
    """Return a configured chat LLM instance for the selected provider.

    Return type is BaseChatModel — concrete subclass depends on provider
    (ChatOpenAI for OpenAI-compatible endpoints, ChatAnthropic for Claude).
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Supported: {', '.join(_PROVIDERS)}."
        )

    cfg = _PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"])
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError(
            f"{cfg['env_key']} is missing/placeholder in .env. "
            f"Set it to a real API key from provider '{provider}'."
        )

    model = model or os.getenv("LLM_MODEL") or cfg["default_model"]

    if cfg["flavour"] == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError(
                "langchain-anthropic is not installed. Run: "
                "pip install langchain-anthropic"
            ) from e
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=api_key,
        )

    if cfg["flavour"] == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError(
                "langchain-google-genai is not installed. Run: "
                "pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            google_api_key=api_key,
        )

    # openai_compat flavour
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
        "api_key": api_key,
    }
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return ChatOpenAI(**kwargs)


def current_provider_summary() -> str:
    """Human-readable string for logging."""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if provider not in _PROVIDERS:
        return f"provider={provider} (invalid)"
    cfg = _PROVIDERS[provider]
    model = os.getenv("LLM_MODEL") or cfg["default_model"]
    key_set = bool(os.getenv(cfg["env_key"])) and not (
        os.getenv(cfg["env_key"], "").startswith("your_")
    )
    return f"{provider} / {model} / key {'OK' if key_set else 'MISSING'}"
