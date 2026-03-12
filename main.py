import os
import asyncio
from app.tracing.client_telemetry import TracingClient
from agents import Agent, Runner

async def start_app():
    client = TracingClient()
    
    if client.enabled:
        client.instrument()
        print("OTel Instrumentation active")

    if client.enabled:
        with client.start_span(name="agent-run") as span:
            try:
                agent = Agent(
                    name="test-agent",
                    instructions="You are a helpful assistant.",
                )
                result = await Runner.run(agent, "Say Hello in a sentence")
                span.update(output=result.final_output)
                print(result.final_output)

            except Exception as exc:
                span.update(
                    level="ERROR",
                    status_message=str(exc),
                )
                span.record_exception(exc)
                print(f"Error captured in span: {exc}")

    await client.async_shutdown()

if __name__ == "__main__":
    asyncio.run(start_app())