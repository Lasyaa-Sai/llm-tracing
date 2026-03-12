"""Tracing decorators for execution tracing.

Drop-in replacement for the previous Langfuse-SDK-based decorators.
All behaviour is preserved; the only change is that the underlying
tracing client is now pure OpenTelemetry (see client.py).

Public API (unchanged):
  @trace_execution(name="...")   — wraps an async method as a root trace or
                                   child span depending on context.
  @trace_llm_generation          — wraps an LLM execute() call as a generation
                                   span capturing model, tokens, and I/O.
"""

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
    return TracingClient()


def trace_execution(
    name: str | None = None,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    """Decorator to wrap async method execution with tracing.

    Behaviour is identical to the previous Langfuse-SDK version:

    * When called **inside** an existing trace → creates a child span.
      Records ``result.edge`` as the output (if present) and marks the span
      ERROR on exception.

    * When called **outside** any trace → creates a root trace span.
      Sets user_id, session_id, metadata, and tags from ``self._ctx``, then
      records the serialised result as the trace output.

    The span name defaults to ``self.name`` if available, otherwise
    ``func.__name__``.

    Example::

        from app.config.globals import TRACE_HANDLE_MESSAGE

        @trace_execution(name=TRACE_HANDLE_MESSAGE)
        async def execute(self, message: str) -> None:
            ...
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
                    logger.warning("Span setup failed — executing without tracing")
                    return await func(*args, **kwargs)

                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    try:
                        span.update(level="ERROR", status_message=str(exc))
                        span.record_exception(exc)
                        span_ctx.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        logger.warning("Span error recording failed")
                    raise

                try:
                    output = result.edge if hasattr(result, "edge") else None 
                    if output is not None:
                        span.update(output=output)
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    logger.warning("Span finalization failed")

                return result

         
            if ctx is None:
                return await func(*args, **kwargs)

            trace_name = name or getattr(self_arg, "name", None) or func.__name__
            input_data = args[1] if len(args) > 1 else None
            user_id = ctx.user.email if ctx.is_authenticated else None

            try:
                obs_ctx = client.start_observation(name=trace_name, input=input_data)
                obs = obs_ctx.__enter__()
            except Exception:
                logger.warning("Root trace setup failed — executing without tracing")
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
                    obs.record_exception(exc)
                    obs_ctx.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    logger.warning("Root trace error recording failed")
                raise

            try:
                serialized = (
                    result.model_dump() if isinstance(result, BaseModel) else result
                )
                obs.update_trace(output=serialized)
                obs_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("Root trace finalization failed")

            return result

        return wrapper

    return decorator



def trace_llm_generation[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Decorator to trace an LLM ``execute()`` call as a generation span.

    Behaviour is identical to the previous Langfuse-SDK version:

    * Only traces when inside an existing parent trace.
    * Captures model name, temperature, response_format, system/user
      messages, output text, and token usage from the returned ``LlmResult``.
    * If tracing is disabled or there is no active parent, the function is
      called directly with no overhead.

    Usage::

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

        try:
            parent_id = client.get_current_observation_id()
        except Exception:
            return await func(*args, **kwargs)
        if parent_id is None:
            return await func(*args, **kwargs)

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        self_arg = bound.arguments.get("self")
        model_id: str | None = getattr(self_arg, "_model_id", None)
        temperature: float | None = getattr(self_arg, "_temperature", None)
        system_instructions: str | None = bound.arguments.get("system_instructions")
        user_input: str | None = bound.arguments.get("user_input")
        output_schema: type[BaseModel] | None = bound.arguments.get("output_schema")

        input_messages: list[dict[str, str]] = []
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
                input=input_messages or None,
                model_parameters=model_parameters or None,
            )
            generation = gen_ctx.__enter__()
        except Exception:
            logger.warning("Generation span setup failed — executing without tracing")
            return await func(*args, **kwargs)

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            try:
                generation.update(level="ERROR", status_message=str(exc))
                generation.record_exception(exc)
                gen_ctx.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.warning("Generation error recording failed")
            raise

        try:
            output = getattr(result, "output", None)
            usage = getattr(result, "usage", None)

            serialized_output = (
                (output.model_dump() if isinstance(output, BaseModel) else output)
                if output is not None
                else None
            )

            usage_details: dict[str, int] | None = None
            if usage is not None:
                usage_details = {
                    "input": (
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", 0)
                        or 0
                    ),
                    "output": (
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", 0)
                        or 0
                    ),
                }

            generation.update(output=serialized_output, usage_details=usage_details)
            gen_ctx.__exit__(None, None, None)
        except Exception:
            logger.warning("Generation finalization failed")

        return result

    return wrapper
