"""Singleton tracing client — the only module that imports from ``langfuse``."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

from app.config.globals import (
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from app.core.logging import logger
from app.core.singleton import SingletonMeta


class TracingClient(metaclass=SingletonMeta):
    """Application-wide tracing client backed by Langfuse.

    Uses :class:`SingletonMeta` so that ``TracingClient()`` always returns the
    same instance.  All Langfuse / openinference imports are confined to this
    class so the rest of the codebase stays decoupled.

    Callers **must** check :attr:`enabled` before calling observation methods
    (``start_span``, ``start_observation``, ``start_generation``,
    ``get_current_observation_id``), as those require an active client.
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        self._instrumentor_active = False

        if not LANGFUSE_ENABLED:
            logger.info("Langfuse tracing disabled")
            return

        if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse keys are empty — tracing disabled")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_BASE_URL,
            )
        except ImportError:
            logger.warning(
                "Langfuse package not installed. Run: poetry install --extras tracing"
            )
        except Exception as e:
            logger.warning(f"Langfuse client init failed ({type(e).__name__}): {e}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return ``True`` when a Langfuse client is available."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Instrumentation (absorbs _init_langfuse from app.py)
    # ------------------------------------------------------------------

    def instrument(self) -> None:
        """Activate OpenAI Agents instrumentation (idempotent)."""
        if not self.enabled or self._instrumentor_active:
            return

        try:
            from openinference.instrumentation.openai_agents import (
                OpenAIAgentsInstrumentor,
            )

            OpenAIAgentsInstrumentor().instrument()
            self._instrumentor_active = True
            logger.info("Langfuse tracing enabled")
        except ImportError as e:
            logger.warning(
                f"Langfuse tracing unavailable — missing package: {e}. "
                "Run: poetry install --extras tracing"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse instrumentation: {e}")

    # ------------------------------------------------------------------
    # Observation context managers
    # ------------------------------------------------------------------

    @contextmanager
    def start_span(self, *, name: str) -> Any:
        """Wrap ``_client.start_as_current_span``."""
        assert self._client is not None
        with self._client.start_as_current_span(name=name) as span:
            yield span

    @contextmanager
    def start_observation(self, *, name: str, input: Any = None) -> Any:
        """Wrap ``_client.start_as_current_observation``."""
        assert self._client is not None
        with self._client.start_as_current_observation(name=name, input=input) as obs:
            yield obs

    @contextmanager
    def start_generation(
        self,
        *,
        name: str,
        model: str | None,
        input: Any = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Any:
        """Wrap ``_client.start_as_current_generation``."""
        assert self._client is not None
        with self._client.start_as_current_generation(
            name=name,
            model=model,
            input=input,
            model_parameters=model_parameters,
        ) as gen:
            yield gen

    def get_current_observation_id(self) -> str | None:
        """Return the current observation ID, or ``None``."""
        assert self._client is not None
        return self._client.get_current_observation_id()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush pending events and shut down the client."""
        if self._client is None:
            return
        self._client.flush()
        self._client.shutdown()
        self._client = None

    async def async_shutdown(self) -> None:
        """Non-blocking shutdown — runs :meth:`shutdown` in an executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.shutdown)
