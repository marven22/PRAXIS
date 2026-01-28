"""Optional GPT integrations and prompt building."""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .programs import GPT_TEMPLATES


def _truncate(s: str, n: int = 1200) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"... [truncated {len(s) - n} chars]"


def gpt_build_prompt(context: Dict[str, Any], num_classes: int) -> str:
    templates_desc = [
        {"template": k, "desc": v["desc"], "params": v["params"]}
        for k, v in GPT_TEMPLATES.items()
    ]

    prompt_obj = {
        "task": "Propose a small set of symbolic programs (hard-label rules) to add to a PRAXIS archive.",
        "constraints": [
            "You must ONLY use the provided templates; do NOT invent new templates.",
            "Return STRICT JSON only, no markdown, no commentary.",
            "Return an object with key 'programs' mapping to a list of program specs.",
            "Each program spec must have: template (string), params (object), name (string, optional), family='gpt'.",
            "Keep it diverse: propose programs that differ in template and parameters.",
            "Suggest at most N programs as requested.",
            f"Classes are indexed 0..{num_classes - 1}.",
        ],
        "available_templates": templates_desc,
        "current_context": context,
        "output_schema": {
            "programs": [
                {"template": "parity_bias", "params": {"parity": 0}, "name": "optional", "family": "gpt"}
            ]
        },
    }
    return json.dumps(prompt_obj, indent=2)


def _load_openai_module() -> Tuple[Optional[Any], Optional[str]]:
    spec = importlib.util.find_spec("openai")
    if spec is None:
        return None, "openai package not installed"
    module = importlib.import_module("openai")
    return module, None


def gpt_call_openai(prompt: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY not set", "raw": ""}

    module, err = _load_openai_module()
    if err or module is None:
        return {"ok": False, "error": err or "openai import failed", "raw": ""}

    try:
        client = module.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful JSON-only assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return {"ok": True, "raw": text}
    except Exception as exc:
        return {"ok": False, "error": f"openai_call_failed: {type(exc).__name__}: {exc}", "raw": ""}


def gpt_call_azure(prompt: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    if not api_key or not endpoint:
        return {
            "ok": False,
            "error": "AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set",
            "raw": "",
        }

    module, err = _load_openai_module()
    if err or module is None:
        return {"ok": False, "error": err or "openai import failed", "raw": ""}

    try:
        client = module.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful JSON-only assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return {"ok": True, "raw": text}
    except Exception as exc:
        return {"ok": False, "error": f"azure_openai_call_failed: {type(exc).__name__}: {exc}", "raw": ""}


def gpt_suggest_programs(
    *,
    enabled: bool,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_new_programs: int,
    context: Dict[str, Any],
    num_classes: int,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "ok": True,
            "status": "disabled",
            "prompt": "",
            "raw": "",
            "parsed_program_specs": [],
            "compiled": [],
            "errors": [],
        }

    provider = (provider or "openai").strip().lower()
    prompt = gpt_build_prompt(context=context, num_classes=num_classes)
    if provider == "openai":
        call_res = gpt_call_openai(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "azure":
        call_res = gpt_call_azure(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return {
            "enabled": True,
            "ok": False,
            "status": f"unknown_provider:{provider}",
            "prompt": _truncate(prompt),
            "raw": "",
            "parsed_program_specs": [],
            "compiled": [],
            "errors": [f"Unknown provider: {provider}"],
        }

    raw = call_res.get("raw", "") if isinstance(call_res, dict) else ""
    if not call_res.get("ok", False):
        return {
            "enabled": True,
            "ok": False,
            "status": "call_failed",
            "prompt": _truncate(prompt),
            "raw": _truncate(raw),
            "parsed_program_specs": [],
            "compiled": [],
            "errors": [call_res.get("error", "unknown_error")],
        }

    parsed_specs: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "programs" not in obj:
            raise ValueError("JSON must be an object with key 'programs'")
        progs = obj.get("programs", [])
        if not isinstance(progs, list):
            raise ValueError("'programs' must be a list")
        progs = progs[: int(max_new_programs)]
        for sp in progs:
            if not isinstance(sp, dict):
                continue
            sp2 = {
                "template": sp.get("template", ""),
                "params": sp.get("params", {}) if isinstance(sp.get("params", {}), dict) else {},
                "name": sp.get("name", ""),
                "family": "gpt",
            }
            parsed_specs.append(sp2)
    except Exception as exc:
        errors.append(f"json_parse_failed: {type(exc).__name__}: {exc}")

    return {
        "enabled": True,
        "ok": len(errors) == 0,
        "status": "ok" if len(errors) == 0 else "parse_failed",
        "prompt": _truncate(prompt),
        "raw": _truncate(raw),
        "parsed_program_specs": parsed_specs,
        "compiled": [],
        "errors": errors,
    }
