
from __future__ import annotations

import asyncio
import base64
import json
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import baggage, context, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan, Span, StatusCode
from app.config.globals import LANGFUSE_ENABLED
from app.core.logging import logger
from app.core.singleton import SingletonMeta


class _SpanWrapper:
    """Thin wrapper around an OTel Span that mirrors the subset of the
    Langfuse observation/generation API that the decorators rely on.

    Methods
    -------
    update(**kwargs)
        Accepts any combination of:
          level, status_message, output, usage_details, input
        and translates them to the correct Langfuse OTel attributes.
    update_trace(**kwargs)
        Sets trace-level attributes (user_id, session_id, metadata, tags,
        output) on the span so Langfuse attaches them to the root trace.
    """

    def __init__(self, span: Span) -> None:
        self._span = span


    def update(
        self,
        *,
        level: str | None = None,
        status_message: str | None = None,
        output: Any = None,
        usage_details: dict[str, Any] | None = None,
        input: Any = None,
    ) -> None:
        span = self._span

        if level == "ERROR":
            span.set_status(StatusCode.ERROR, status_message or "")
            if status_message:
                span.set_attribute(
                    "langfuse.observation.status_message", status_message
                )
        elif status_message:
            span.set_attribute(
                "langfuse.observation.status_message", status_message
            )

        if output is not None:
            serialized = (
                output
                if isinstance(output, str)
                else json.dumps(output, default=str)
            )
        
            span.set_attribute("langfuse.observation.output", serialized)

        if input is not None:
            serialized_in = (
                input
                if isinstance(input, str)
                else json.dumps(input, default=str)
            )
            span.set_attribute("langfuse.observation.input", serialized_in)

        if usage_details is not None:
            input_tokens = usage_details.get("input", 0) or 0
            output_tokens = usage_details.get("output", 0) or 0
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)


    def update_trace(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        output: Any = None,
    ) -> None:
        span = self._span

        if user_id is not None:
            # Langfuse maps langfuse.user.id → trace.userId
            span.set_attribute("langfuse.user.id", user_id)

        if session_id is not None:
            # Langfuse maps langfuse.session.id → trace.sessionId
            span.set_attribute("langfuse.session.id", session_id)

        if metadata:
            for key, value in metadata.items():
                attr_val = (
                    value if isinstance(value, str) else json.dumps(value, default=str)
                )
                span.set_attribute(f"langfuse.trace.metadata.{key}", attr_val)

        if tags is not None:
            span.set_attribute("langfuse.trace.tags", tags)

        if output is not None:
            serialized = (
                output
                if isinstance(output, str)
                else json.dumps(output, default=str)
            )
            span.set_attribute("langfuse.trace.output", serialized)
    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)


class TracingClient(metaclass=SingletonMeta):
    """Exports spans to Fluent Bit via OTLP HTTP (localhost:4318).
Fluent Bit handles forwarding to Langfuse.
    """

    def __init__(self) -> None:
        self._tracer: trace.Tracer | None = None
        self._provider: TracerProvider | None = None
        self._instrumentor_active = False

        if not LANGFUSE_ENABLED:
            logger.info("Tracing disabled (LANGFUSE_ENABLED=false)")
            return

        try:
            exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
)

            self._provider = TracerProvider()
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(self._provider)

            self._tracer = trace.get_tracer(__name__)
            logger.info("OTel → Fluent Bit tracing initialised")

        except ImportError:
            logger.warning(
                "opentelemetry packages not installed. "
                "Run: poetry install --extras tracing"
            )
        except Exception as exc:
            logger.warning("OTel tracing init failed (%s): %s", type(exc).__name__, exc)


    @property
    def enabled(self) -> bool:
        """Return ``True`` when a tracer is available."""
        return self._tracer is not None


    def instrument(self) -> None:
   
        if not self.enabled or self._instrumentor_active:
            return

        try:
            from openinference.instrumentation.openai_agents import (
                OpenAIAgentsInstrumentor,
            )

            OpenAIAgentsInstrumentor().instrument()
            self._instrumentor_active = True
            logger.info("OpenAI Agents auto-instrumentation active")
        except ImportError as exc:
            logger.warning(
                "Tracing unavailable — missing package: %s. "
                "Run: poetry install --extras tracing",
                exc,
            )
        except Exception as exc:
            logger.warning("Failed to initialise OpenAI Agents instrumentation: %s", exc)


    def get_current_observation_id(self) -> str | None:

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is not None and ctx.is_valid and not isinstance(span, NonRecordingSpan):
            return format(ctx.span_id, "016x")
        return None

    @contextmanager
    def start_span(self, *, name: str) -> Generator[_SpanWrapper, None, None]:
        """Create a child span, yield a :class:`_SpanWrapper`."""
        assert self._tracer is not None
        with self._tracer.start_as_current_span(name) as span:
            yield _SpanWrapper(span)

    @contextmanager
    def start_observation(
        self, *, name: str, input: Any = None
    ) -> Generator[_SpanWrapper, None, None]:
        assert self._tracer is not None
        with self._tracer.start_as_current_span(name) as span:
            wrapper = _SpanWrapper(span)
            if input is not None:
                wrapper.update(input=input)
            yield wrapper

    @contextmanager
    def start_generation(
        self,
        *,
        name: str,
        model: str | None,
        input: Any = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create a generation span (LLM call), yield a :class:`_SpanWrapper`.

        Sets the GenAI semantic-convention attributes that Langfuse uses to
        identify the span as a *generation* and to populate the model,
        temperature, and input fields in the trace detail view.

        Attribute mapping reference:
        https://langfuse.com/integrations/native/opentelemetry#observation-level-attributes
        """
        assert self._tracer is not None
        with self._tracer.start_as_current_span(name) as span:
            
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("gen_ai.operation.name", "chat")

            if model:
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("langfuse.observation.model.name", model)

            if input is not None:
                serialized = (
                    input
                    if isinstance(input, str)
                    else json.dumps(input, default=str)
                )
                span.set_attribute("langfuse.observation.input", serialized)

            if model_parameters:
                temp = model_parameters.get("temperature")
                if temp is not None:
                    span.set_attribute("gen_ai.request.temperature", float(temp))

                response_format = model_parameters.get("response_format")
                if response_format is not None:
                    params_json = json.dumps(model_parameters, default=str)
                    span.set_attribute(
                        "langfuse.observation.model.parameters", params_json
                    )

            yield _SpanWrapper(span)

    def shutdown(self) -> None:
        
        if self._provider is None:
            return
        try:
            self._provider.shutdown()
        except Exception as exc:
            logger.warning("OTel provider shutdown error: %s", exc)
        finally:
            self._provider = None
            self._tracer = None

    async def async_shutdown(self) -> None:
    
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.shutdown)
