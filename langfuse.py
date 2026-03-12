"""Langfuse tracing decorators for execution tracing."""

import inspect
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel

from app.config.globals import API_TYPE
from app.core.logging import logger
from app.services.tracing.client import TracingClient

P = ParamSpec("P")
R = TypeVar("R")


def _get_client() -> TracingClient:
    """Return the singleton :class:`TracingClient` (easy to patch in tests)."""
    return TracingClient()


def trace_execution(
    name: str | None = None,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    """Decorator to wrap async method execution with Langfuse tracing.

    When no parent trace exists, creates a new top-level trace with
    user_id, session_id, and metadata from ctx. When called inside an
    existing trace, creates a child span instead (no flush, no metadata).

    The span name defaults to self.name if available, otherwise
    func.__name__.

    Args:
        name: Optional trace/span name override.

    Returns:
        Decorated async function with Langfuse tracing.

    Example:
        from app.config.globals import TRACE_HANDLE_MESSAGE

        @trace_execution(name=TRACE_HANDLE_MESSAGE)
        async def execute(self, message: str) -> None:
            # Implementation
            pass
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            client = _get_client()
            if not client.enabled:
                return await func(*args, **kwargs)

            self_arg = args[0] if args else None
            ctx = getattr(self_arg, "_ctx", None) if self_arg else None

            # If already inside a trace, create a child span
            try:
                parent_id = client.get_current_observation_id()
            except Exception:
                return await func(*args, **kwargs)
            if parent_id is not None:
                span_name = name or getattr(self_arg, "name", None) or func.__name__
                try:
                    span_ctx = client.start_span(name=span_name)
                    span = span_ctx.__enter__()
                except Exception:
                    logger.warning(
                        "Langfuse span setup failed — executing without tracing"
                    )
                    return await func(*args, **kwargs)

                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    try:
                        span.update(level="ERROR", status_message=str(exc))
                        span_ctx.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        logger.warning("Langfuse span error recording failed")
                    raise

                try:
                    output = result.edge if hasattr(result, "edge") else None  # type: ignore[union-attr]
                    if output is not None:
                        span.update(output=output)
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    logger.warning("Langfuse span finalization failed")
                return result

            # No parent — create a new top-level trace
            if ctx is None:
                return await func(*args, **kwargs)

            trace_name = name or getattr(self_arg, "name", None) or func.__name__
            input_data = args[1] if len(args) > 1 else None
            user_id = ctx.user.email if ctx.is_authenticated else None
            try:
                obs_ctx = client.start_observation(name=trace_name, input=input_data)
                obs = obs_ctx.__enter__()
            except Exception:
                logger.warning(
                    "Langfuse trace setup failed — executing without tracing"
                )
                return await func(*args, **kwargs)

            try:
                obs.update_trace(
                    user_id=user_id,
                    session_id=ctx.conversation_id,
                    metadata={"request_id": ctx.request_id},
                    tags=[API_TYPE.value],
                )
            except Exception:
                pass

            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                try:
                    obs.update(level="ERROR", status_message=str(exc))
                    obs_ctx.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    logger.warning("Langfuse trace error recording failed")
                raise

            try:
                serialized = (
                    result.model_dump() if isinstance(result, BaseModel) else result
                )
                obs.update_trace(output=serialized)
                obs_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("Langfuse trace finalization failed")
            return result

        return wrapper

    return decorator


def trace_llm_generation[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Decorator to trace an LLM execute() call as a Langfuse generation.

    Creates a generation span as a child of the current trace (if one exists),
    capturing the model name, input (system_instructions + user_input), output
    text, and token usage from the returned LlmResult.

    If tracing is disabled or there is no active parent trace, the function is
    called directly with no overhead.

    Usage:
        @trace_llm_generation
        async def execute(self, system_instructions, user_input, ...) -> LlmResult:
            ...
    """
    sig = inspect.signature(func)

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        client = _get_client()
        if not client.enabled:
            return await func(*args, **kwargs)

        # Only trace when inside an existing parent trace
        try:
            parent_id = client.get_current_observation_id()
        except Exception:
            return await func(*args, **kwargs)
        if parent_id is None:
            return await func(*args, **kwargs)

        # Bind args to param names so we can extract inputs by name.
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        self_arg = bound.arguments.get("self")
        model_id: str | None = getattr(self_arg, "_model_id", None)
        temperature: float | None = getattr(self_arg, "_temperature", None)
        system_instructions: str | None = bound.arguments.get("system_instructions")
        user_input: str | None = bound.arguments.get("user_input")
        output_schema: type[BaseModel] | None = bound.arguments.get("output_schema")

        input_messages = []
        if system_instructions:
            input_messages.append({"role": "system", "content": system_instructions})
        if user_input:
            input_messages.append({"role": "user", "content": user_input})

        model_parameters: dict[str, Any] = {}
        if temperature is not None:
            model_parameters["temperature"] = temperature
        if output_schema is not None:
            model_parameters["response_format"] = output_schema.__name__

        try:
            gen_ctx = client.start_generation(
                name=model_id or func.__name__,
                model=model_id,
                input=input_messages,
                model_parameters=model_parameters or None,
            )
            generation = gen_ctx.__enter__()
        except Exception:
            logger.warning(
                "Langfuse generation setup failed — executing without tracing"
            )
            return await func(*args, **kwargs)

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            try:
                generation.update(level="ERROR", status_message=str(exc))
                gen_ctx.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.warning("Langfuse generation error recording failed")
            raise

        # LlmResult has .output (str | BaseModel) and .usage
        try:
            output = getattr(result, "output", None)
            usage = getattr(result, "usage", None)
            serialized_output = (
                (output.model_dump() if isinstance(output, BaseModel) else output)
                if output is not None
                else None
            )
            generation.update(
                output=serialized_output,
                usage_details={
                    "input": getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "input_tokens", 0)
                    or 0,
                    "output": getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", 0)
                    or 0,
                }
                if usage is not None
                else None,
            )
            gen_ctx.__exit__(None, None, None)
        except Exception:
            logger.warning("Langfuse generation finalization failed")
        return result

    return wrapper
