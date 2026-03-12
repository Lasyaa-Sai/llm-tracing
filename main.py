import asyncio
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from agents import Agent, Runner
from app.tracing.client_telemetry import TracingClient

async def start_app():
    client = TracingClient()
    
    if client.enabled:
        client.instrument()
        print(" OTel Instrumentation active")

    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
    )

    result = await Runner.run(agent, "Say Hello in a sentence,Tell me interesting facts about earth")
    print(result.final_output)

    await client.async_shutdown()

if __name__ == "__main__":
    asyncio.run(start_app())