#!/usr/bin/env python3
"""
Mixture-of-Agents Tool Module

This module implements the Mixture-of-Agents (MoA) methodology that leverages
the collective strengths of multiple LLMs through a layered architecture to
achieve state-of-the-art performance on complex reasoning tasks.

Based on the research paper: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692v1)

Key Features:
- Multi-layer LLM collaboration for enhanced reasoning
- Parallel processing of reference models for efficiency
- Intelligent aggregation and synthesis of diverse responses
- Specialized for extremely difficult problems requiring intense reasoning
- Optimized for coding, mathematics, and complex analytical tasks

Available Tool:
- mixture_of_agents_tool: Process complex queries using multiple frontier models

Architecture:
1. Reference models generate diverse initial responses in parallel
2. Aggregator model synthesizes responses into a high-quality output
3. Multiple layers can be used for iterative refinement (future enhancement)

Models Used:
- Reference Models: configurable via HERMES_MOA_REFERENCE_MODELS
- Aggregator Model: configurable via HERMES_MOA_AGGREGATOR_MODEL

Configuration:
    To customize the MoA setup, modify the configuration constants at the top of this file:
    - REFERENCE_MODELS: List of models for generating diverse initial responses
    - AGGREGATOR_MODEL: Model used to synthesize the final response
    - REFERENCE_TEMPERATURE/AGGREGATOR_TEMPERATURE: Sampling temperatures
    - MIN_SUCCESSFUL_REFERENCES: Minimum successful models needed to proceed

Usage:
    from mixture_of_agents_tool import mixture_of_agents_tool
    import asyncio
    
    # Process a complex query
    result = await mixture_of_agents_tool(
        user_prompt="Solve this complex mathematical proof..."
    )
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from tools.openrouter_client import get_async_client as _get_openrouter_client, check_api_key as check_openrouter_api_key
from agent.auxiliary_client import extract_content_or_reasoning
from tools.debug_helpers import DebugSession
import sys

logger = logging.getLogger(__name__)

# Configuration for MoA processing
def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()] or default


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_boolish(name: str, default: str = "auto") -> str:
    return os.getenv(name, default).strip().lower() or default


GEMINI_REFERENCE_MODEL = (
    os.getenv("HERMES_MOA_GEMINI_REFERENCE_MODEL", "gemini/gemini-3.5-flash").strip()
    or "gemini/gemini-3.5-flash"
)
GEMINI_OAUTH_REFERENCE_MODEL = (
    os.getenv("HERMES_MOA_GEMINI_OAUTH_REFERENCE_MODEL", "google-gemini-cli/gemini-3.5-flash").strip()
    or "google-gemini-cli/gemini-3.5-flash"
)


def _gemini_reference_defaults() -> List[str]:
    """Return a Gemini reference only when it is likely usable.

    The default is "auto": include Google AI Studio when GOOGLE_API_KEY or
    GEMINI_API_KEY is present, or Gemini CLI OAuth when Hermes is logged in.
    This avoids adding an unavailable model that would slow every MoA call.
    """
    mode = _env_boolish("HERMES_MOA_ENABLE_GEMINI_REFERENCE", "auto")
    if mode in {"0", "false", "no", "off", "disabled"}:
        return []
    if mode in {"1", "true", "yes", "on", "force", "forced"}:
        return [GEMINI_REFERENCE_MODEL]
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return [GEMINI_REFERENCE_MODEL]
    try:
        from hermes_cli.auth import get_gemini_oauth_auth_status

        if (get_gemini_oauth_auth_status() or {}).get("logged_in"):
            return [GEMINI_OAUTH_REFERENCE_MODEL]
    except Exception:
        pass
    return []


# Reference models generate independent, tool-less analysis in parallel.
# `custom/opus` routes to cabinlab Claude Max; `xai-oauth/grok-4.3` routes to
# Hermes' xAI OAuth provider; `openai-codex/gpt-5.5` routes to Codex.
# Gemini 3.5 Flash is appended automatically only when credentials exist.
# OpenRouter slugs still work when passed explicitly.
REFERENCE_MODELS = _env_list(
    "HERMES_MOA_REFERENCE_MODELS",
    [
        "custom/opus",
        "openai-codex/gpt-5.5",
        "xai-oauth/grok-4.3",
        *_gemini_reference_defaults(),
    ],
)

# Aggregator model synthesizes the reference responses. Keep this on GPT-5.5 by
# default so the aggregation prompt can be rich without affecting Claude's Max
# subscription prompt-size behavior.
AGGREGATOR_MODEL = (
    os.getenv("HERMES_MOA_AGGREGATOR_MODEL", "openai-codex/gpt-5.5").strip()
    or "openai-codex/gpt-5.5"
)

# Temperature settings optimized for MoA performance.
REFERENCE_TEMPERATURE = 0.6
AGGREGATOR_TEMPERATURE = 0.4

# Failure handling configuration.
MIN_SUCCESSFUL_REFERENCES = int(os.getenv("HERMES_MOA_MIN_SUCCESSFUL_REFERENCES", "1"))

DEFAULT_REFERENCE_SYSTEM_PROMPT = """You are a reference model inside a Mixture-of-Agents process.

Analyze the user's request independently. Prefer English for internal analysis and dense technical substance, even when the user writes in another language. Preserve all user-specified Japanese wording, names, constraints, commands, paths, and requested output language exactly.

Do not try to be polished. Do not summarize the task back. Focus on:
- correctness and hidden assumptions
- edge cases and failure modes
- concrete implementation or verification risks
- alternative approaches and tradeoffs
- concise evidence the aggregator should consider

You do not have tools in this reference call. If tool use would be needed, state precisely what should be checked rather than pretending it was checked."""

REFERENCE_SYSTEM_PROMPT = _env_text(
    "HERMES_MOA_REFERENCE_SYSTEM_PROMPT",
    DEFAULT_REFERENCE_SYSTEM_PROMPT,
)

# Rich/non-compact aggregator prompt. This only goes to the aggregator model
# inside MoA; it is not the outer Hermes system prompt.
DEFAULT_AGGREGATOR_SYSTEM_PROMPT = """You are the aggregator in a Mixture-of-Agents process.

You will receive independent reference responses to the latest user request. Synthesize them into the final answer. Optimize for truth, usefulness, and actionability rather than consensus. Explicitly resolve disagreements, discard weak claims, and preserve the strongest mechanisms, caveats, and verification steps.

Rules:
- Preserve the user's requested language for the final answer unless the user asks otherwise.
- If the user wrote Japanese, produce natural Japanese in the final answer, while keeping code, commands, paths, identifiers, and quoted strings exact.
- Do not mention that you are an aggregator unless it is directly useful.
- Do not include a generic summary section. Give the answer or the next concrete action.
- Prefer compact but information-dense output.
- If references lack evidence, say what remains unverified instead of overstating certainty.

Reference responses:"""

AGGREGATOR_SYSTEM_PROMPT = _env_text(
    "HERMES_MOA_AGGREGATOR_SYSTEM_PROMPT",
    DEFAULT_AGGREGATOR_SYSTEM_PROMPT,
)


def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    response_text = "\n\n".join(
        f"Reference response {i + 1}:\n{response}"
        for i, response in enumerate(responses)
    )
    return f"{system_prompt}\n\n{response_text}"


def _model_route(model: str) -> tuple[str, str]:
    """Return (provider, model_slug) for a MoA model spec."""
    value = (model or "").strip()
    if value.startswith("custom/"):
        return "custom", value.split("/", 1)[1] or "opus"
    if value in {"opus", "sonnet", "haiku"}:
        return "custom", value
    if value.startswith("openai-codex/"):
        return "openai-codex", value.split("/", 1)[1] or "gpt-5.5"
    if value.startswith("codex/"):
        return "openai-codex", value.split("/", 1)[1] or "gpt-5.5"
    if value.startswith("xai-oauth/"):
        return "xai-oauth", value.split("/", 1)[1] or "grok-4.3"
    if value.startswith("xai/"):
        return "xai", value.split("/", 1)[1] or "grok-4.3"
    if value.startswith("grok/"):
        return "xai-oauth", value.split("/", 1)[1] or "grok-4.3"
    if value.startswith("google-gemini-cli/"):
        return "google-gemini-cli", value.split("/", 1)[1] or "gemini-3.5-flash"
    if value.startswith("gemini-cli/"):
        return "google-gemini-cli", value.split("/", 1)[1] or "gemini-3.5-flash"
    if value.startswith("gemini/"):
        return "gemini", value.split("/", 1)[1] or "gemini-3.5-flash"
    if value.startswith("google/"):
        return "gemini", value.split("/", 1)[1] or "gemini-3.5-flash"
    return "openrouter", value


def _reasoning_extra_body(model: str) -> Optional[Dict[str, Any]]:
    provider, _ = _model_route(model)
    if provider != "openrouter":
        return None
    return {"reasoning": {"enabled": True, "effort": "xhigh"}}


def _moa_client_for_model(model: str):
    provider, routed_model = _model_route(model)
    if provider in {"custom", "openai-codex", "xai-oauth", "xai", "gemini", "google-gemini-cli"}:
        from agent.auxiliary_client import resolve_provider_client

        client, resolved = resolve_provider_client(provider, model=routed_model, async_mode=True)
        if client is None:
            raise ValueError(f"Hermes provider {provider!r} is not configured for MoA")
        return client, resolved or routed_model
    return _get_openrouter_client(), routed_model


MOA_CONTEXT_MAX_CHARS = _env_int("HERMES_MOA_CONTEXT_MAX_CHARS", 8000)
MOA_CONTEXT_MAX_MESSAGES = _env_int("HERMES_MOA_CONTEXT_MAX_MESSAGES", 24)


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit < 32:
        return text[:limit]
    head = max(1, limit // 2)
    tail = max(1, limit - head - 15)
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _recent_session_context(session_id: Optional[str]) -> tuple[str, int, int]:
    """Load bounded recent user/assistant context for MoA reference calls."""
    if not session_id or MOA_CONTEXT_MAX_CHARS <= 0 or MOA_CONTEXT_MAX_MESSAGES <= 0:
        return "", 0, 0
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB(read_only=True)
        rows = db.get_messages(session_id)
    except Exception as exc:
        logger.debug("MoA could not load session context for %s: %s", session_id, exc, exc_info=True)
        return "", 0, 0
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass

    selected = []
    for row in reversed(rows):
        role = row.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = _stringify_message_content(row.get("content")).strip()
        if not content:
            continue
        selected.append((role, content))
        if len(selected) >= MOA_CONTEXT_MAX_MESSAGES:
            break
    selected.reverse()

    parts = []
    total = 0
    per_message_limit = max(800, MOA_CONTEXT_MAX_CHARS // max(1, min(MOA_CONTEXT_MAX_MESSAGES, len(selected) or 1)))
    for role, content in selected:
        line = f"{role}: {_clip_text(content, per_message_limit)}"
        remaining = MOA_CONTEXT_MAX_CHARS - total
        if remaining <= 0:
            break
        line = _clip_text(line, remaining)
        parts.append(line)
        total += len(line) + 2
    context = "\n\n".join(parts).strip()
    return context, len(parts), len(context)


def _augment_user_prompt_with_context(user_prompt: str, session_id: Optional[str]) -> tuple[str, int, int]:
    context, message_count, char_count = _recent_session_context(session_id)
    if not context:
        return user_prompt, 0, 0
    return (
        "Recent Hermes conversation context (oldest to newest; tool outputs omitted; "
        "use this only as supporting context, and prioritize the latest request):\n\n"
        f"{context}\n\n"
        "Latest user request / MoA problem:\n\n"
        f"{user_prompt}"
    ), message_count, char_count



_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")

async def _run_reference_model_safe(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6
) -> tuple[str, str, bool]:
    """
    Run a single reference model with retry logic and graceful failure handling.
    
    Args:
        model (str): Model identifier to use
        user_prompt (str): The user's query
        temperature (float): Sampling temperature for response generation
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple[str, str, bool]: (model_name, response_content_or_error, success_flag)
    """
    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %s/%s)", model, attempt + 1, max_retries)
            
            client, routed_model = _moa_client_for_model(model)
            api_params = {
                "model": routed_model,
                "messages": [
                    {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }
            extra_body = _reasoning_extra_body(model)
            if extra_body:
                api_params["extra_body"] = extra_body
            
            # GPT models do not reliably support custom temperature values.
            # Check the routed model because provider/model specs may be
            # prefixed (for example, openai-codex/gpt-5.5).
            if not routed_model.lower().startswith('gpt-'):
                api_params["temperature"] = temperature
            
            response = await client.chat.completions.create(**api_params)
            
            content = extract_content_or_reasoning(response)
            if not content:
                # Reasoning-only response — let the retry loop handle it
                logger.warning("%s returned empty content (attempt %s/%s), retrying", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
            logger.info("%s responded (%s characters)", model, len(content))
            return model, content, True
            
        except Exception as e:
            error_str = str(e)
            # Keep retry-path logging concise; full tracebacks are reserved for
            # terminal failure paths so long-running MoA retries don't flood logs.
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", model, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", model, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", model, attempt + 1, error_str)

            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{model} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return model, error_msg, False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    model: str = AGGREGATOR_MODEL,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: int = None
) -> str:
    """
    Run the aggregator model to synthesize the final response.
    
    Args:
        system_prompt (str): System prompt with all reference responses
        user_prompt (str): Original user query
        temperature (float): Focused temperature for consistent aggregation
        max_tokens (int): Maximum tokens in final response
        
    Returns:
        str: Synthesized final response
    """
    logger.info("Running aggregator model: %s", model)
    client, routed_model = _moa_client_for_model(model)

    api_params = {
        "model": routed_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    }
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    extra_body = _reasoning_extra_body(model)
    if extra_body:
        api_params["extra_body"] = extra_body

    if not routed_model.lower().startswith('gpt-'):
        api_params["temperature"] = temperature

    response = await client.chat.completions.create(**api_params)

    content = extract_content_or_reasoning(response)

    # Retry once on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await client.chat.completions.create(**api_params)
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
    return content


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Process a complex query using the Mixture-of-Agents methodology.
    
    This tool leverages multiple frontier language models to collaboratively solve
    extremely difficult problems requiring intense reasoning. It's particularly
    effective for:
    - Complex mathematical proofs and calculations
    - Advanced coding problems and algorithm design
    - Multi-step analytical reasoning tasks
    - Problems requiring diverse domain expertise
    - Tasks where single models show limitations
    
    The MoA approach uses a fixed 2-layer architecture:
    1. Layer 1: Multiple reference models generate diverse responses in parallel (temp=0.6)
    2. Layer 2: Aggregator model synthesizes the best elements into final response (temp=0.4)
    
    Args:
        user_prompt (str): The complex query or problem to solve
        reference_models (Optional[List[str]]): Custom reference models to use
        aggregator_model (Optional[str]): Custom aggregator model to use
    
    Returns:
        str: JSON string containing the MoA results with the following structure:
             {
                 "success": bool,
                 "response": str,
                 "models_used": {
                     "reference_models": List[str],
                     "aggregator_model": str
                 },
                 "processing_time": float
             }
    
    Raises:
        Exception: If MoA processing fails or API key is not set
    """
    start_time = datetime.datetime.now()
    moa_user_prompt, context_message_count, context_chars = _augment_user_prompt_with_context(
        user_prompt,
        session_id,
    )
    
    debug_call_data = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": reference_models or REFERENCE_MODELS,
            "aggregator_model": aggregator_model or AGGREGATOR_MODEL,
            "reference_temperature": REFERENCE_TEMPERATURE,
            "aggregator_temperature": AGGREGATOR_TEMPERATURE,
            "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
            "context_message_count": context_message_count,
            "context_chars": context_chars,
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {}
    }
    
    try:
        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])
        
        # Use provided models or defaults
        ref_models = reference_models or REFERENCE_MODELS
        agg_model = aggregator_model or AGGREGATOR_MODEL
        
        logger.info("Using %s reference models in 2-layer MoA architecture", len(ref_models))
        
        # Layer 1: Generate diverse responses from reference models (with failure handling)
        logger.info("Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(model, moa_user_prompt, REFERENCE_TEMPERATURE)
            for model in ref_models
        ])
        
        # Separate successful and failed responses
        successful_responses = []
        failed_models = []
        
        for model_name, content, success in model_results:
            if success:
                successful_responses.append(content)
            else:
                failed_models.append(model_name)
        
        successful_count = len(successful_responses)
        failed_count = len(failed_models)
        
        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        
        if failed_models:
            logger.warning("Failed models: %s", ', '.join(failed_models))
        
        # Check if we have enough successful responses to proceed
        if successful_count < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {MIN_SUCCESSFUL_REFERENCES} successful responses.")
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        
        # Layer 2: Aggregate responses using the aggregator model
        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        final_response = await _run_aggregator_model(
            aggregator_system_prompt,
            moa_user_prompt,
            agg_model,
            AGGREGATOR_TEMPERATURE
        )
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info("MoA processing completed in %.2f seconds", processing_time)
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model
            }
        }
        
        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]
        
        # Log debug information
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Calculate processing time even for errors
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare error response (minimal fields)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": reference_models or REFERENCE_MODELS,
                "aggregator_model": aggregator_model or AGGREGATOR_MODEL
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """Return True when at least one configured MoA route is available."""
    if check_openrouter_api_key():
        return True
    for spec in [*REFERENCE_MODELS, AGGREGATOR_MODEL]:
        try:
            _moa_client_for_model(spec)
            return True
        except Exception:
            continue
    return False



def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(REFERENCE_MODELS),
        "failure_tolerance": f"{len(REFERENCE_MODELS) - MIN_SUCCESSFUL_REFERENCES}/{len(REFERENCE_MODELS)} models can fail"
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🤖 Mixture-of-Agents Tool Module")
    print("=" * 50)
    
    # Check if API key is available
    api_available = check_openrouter_api_key()
    
    if not api_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")
        print("Get API key at: https://openrouter.ai/")
        sys.exit(1)
    else:
        print("✅ OpenRouter API key found")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print("\n⚙️  Current Configuration:")
    print(f"  🤖 Reference models ({len(config['reference_models'])}): {', '.join(config['reference_models'])}")
    print(f"  🧠 Aggregator model: {config['aggregator_model']}")
    print(f"  🌡️  Reference temperature: {config['reference_temperature']}")
    print(f"  🌡️  Aggregator temperature: {config['aggregator_temperature']}")
    print(f"  🛡️  Failure tolerance: {config['failure_tolerance']}")
    print(f"  📊 Minimum successful models: {config['min_successful_references']}")
    
    # Show debug mode status
    if _debug.active:
        print(f"\n🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/moa_tools_debug_{_debug.session_id}.json")
    else:
        print("\n🐛 Debug mode disabled (set MOA_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from mixture_of_agents_tool import mixture_of_agents_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      result = await mixture_of_agents_tool(")
    print("          user_prompt='Solve this complex mathematical proof...'")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nBest use cases:")
    print("  - Complex mathematical proofs and calculations")
    print("  - Advanced coding problems and algorithm design")
    print("  - Multi-step analytical reasoning tasks")
    print("  - Problems requiring diverse domain expertise")
    print("  - Tasks where single models show limitations")
    
    print("\nPerformance characteristics:")
    print("  - Higher latency due to multiple model calls")
    print("  - Significantly improved quality for complex tasks")
    print("  - Parallel processing for efficiency")
    print(f"  - Optimized temperatures: {REFERENCE_TEMPERATURE} for reference models, {AGGREGATOR_TEMPERATURE} for aggregation")
    print("  - Token-efficient: only returns final aggregated response")
    print("  - Resilient: continues with partial model failures")
    print("  - Configurable: easy to modify models and settings at top of file")
    print("  - State-of-the-art results on challenging benchmarks")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export MOA_TOOLS_DEBUG=true")
    print("  # Debug logs capture all MoA processing steps and metrics")
    print("  # Logs saved to: ./logs/moa_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": "Route a hard problem through multiple frontier LLMs collaboratively. Runs multiple reference models plus one aggregator; use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The complex query or problem to solve using multiple AI models. Recent Hermes session context is automatically prepended when available, so include only task-specific details here."
            },
            "reference_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional reference model list. Defaults to custom/opus, openai-codex/gpt-5.5, xai-oauth/grok-4.3, and Gemini 3.5 Flash when Gemini credentials are available. Use custom/opus for cabinlab Claude Max, xai-oauth/grok-4.3 for Grok OAuth, openai-codex/gpt-5.5 for Codex, gemini/gemini-3.5-flash for Google AI Studio, google-gemini-cli/gemini-3.5-flash for Gemini CLI OAuth, or vendor/model slugs for OpenRouter."
            },
            "aggregator_model": {
                "type": "string",
                "description": "Optional aggregator model. Defaults to openai-codex/gpt-5.5 with a rich non-compact aggregation prompt."
            }
        },
        "required": ["user_prompt"]
    }
}

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(
        user_prompt=args.get("user_prompt", ""),
        reference_models=args.get("reference_models"),
        aggregator_model=args.get("aggregator_model"),
        session_id=kw.get("session_id"),
    ),
    check_fn=check_moa_requirements,
    requires_env=[],
    is_async=True,
    emoji="🧠",
)
