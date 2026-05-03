import os
import time
from typing import Callable, List, Optional, Tuple

import requests


def openrouter_headers(
    api_key: str,
    http_referer: str = "",
    title: str = "",
) -> dict:
    if not api_key or not api_key.strip():
        raise ValueError("OPENROUTER_API_KEY is not set.")
    key = api_key.strip().replace('"', "").replace("'", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if title:
        headers["X-Title"] = title
    return headers


def openrouter_chat(
    messages: List[dict],
    model_override: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 60,
    temperature: float = 0.2,
    default_model: str = "openrouter/free",
    http_referer: str = "",
    title: str = "",
) -> Tuple[str, Optional[str]]:
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    payload = {
        "model": model_override or default_model,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=openrouter_headers(key, http_referer=http_referer, title=title),
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    model_used = data.get("model") or data.get("choices", [{}])[0].get("model")
    return content, model_used


def ollama_post(
    path: str,
    payload: dict,
    *,
    base_url: str = "http://localhost:11434/api",
    timeout: int = 300,
    retries: int = 3,
) -> dict:
    for attempt in range(retries):
        try:
            url = f"{base_url}{path}"
            response = requests.post(url, json=payload, timeout=timeout)
            if response.ok:
                return response.json()
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Ollama error {response.status_code}: {detail}")
        except (requests.exceptions.ConnectionError, RuntimeError):
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise
    return {}


def ollama_chat(
    model: str,
    messages: List[dict],
    *,
    base_url: str = "http://localhost:11434/api",
    timeout: int = 300,
    temperature: float = 0.2,
) -> Tuple[str, Optional[str]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = ollama_post(path="/chat", payload=payload, base_url=base_url, timeout=timeout)
    return data.get("message", {}).get("content", ""), data.get("model")


def build_openrouter_chat_fn(
    *,
    model_override: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 60,
    temperature: float = 0.2,
    default_model: str = "openrouter/free",
    http_referer: str = "",
    title: str = "",
) -> Callable[[List[dict]], Tuple[str, Optional[str]]]:
    def _chat_fn(messages: List[dict]) -> Tuple[str, Optional[str]]:
        return openrouter_chat(
            messages=messages,
            model_override=model_override,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            default_model=default_model,
            http_referer=http_referer,
            title=title,
        )

    return _chat_fn


def build_ollama_chat_fn(
    model: str,
    *,
    base_url: str = "http://localhost:11434/api",
    timeout: int = 300,
    temperature: float = 0.2,
) -> Callable[[List[dict]], Tuple[str, Optional[str]]]:
    def _chat_fn(messages: List[dict]) -> Tuple[str, Optional[str]]:
        return ollama_chat(
            model=model,
            messages=messages,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

    return _chat_fn
