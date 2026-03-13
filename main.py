import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import asyncio
from openai import AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from app.tracing.client_telemetry import TracingClient
from agents import Agent, Runner, OpenAIChatCompletionsModel

async def start_app():
    client = TracingClient()
    
    if client.enabled:
        client.instrument()
        OpenAIInstrumentor().instrument()
        print("OTel Instrumentation active")

    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    agent = Agent(
        name="geography-assistant",
        instructions="You are a helpful geography assistant. Answer questions clearly and concisely.",
        model=OpenAIChatCompletionsModel(
            model="google/gemini-2.0-flash-001",
            openai_client=openai_client,
        ),
    )

    with client.start_observation(name="agent-workflow", input="What is the capital of United Arab Emirates?") as obs:
        obs.update_trace(
            user_id="test-user-123",
            session_id="test-session-abc",
            metadata={"environment": "test", "model": "gemini-2.0-flash"},
            tags=["test", "gemini"],
        )
        result = await Runner.run(agent, "What is the capital of United Arab Emirates?")
        obs.update(output=result.final_output)
        print(f"Output: {result.final_output}")

    await asyncio.sleep(3)
    await client.async_shutdown()

if __name__ == "__main__":
    asyncio.run(start_app())