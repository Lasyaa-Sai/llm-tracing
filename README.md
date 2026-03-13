# LLM Tracing

A lightweight OpenTelemetry-based tracing setup for LLM applications. Traces are collected by **Fluent Bit** and forwarded to **Langfuse** — your app has zero Langfuse awareness.

## Architecture

```
Your App (OpenAI Agents SDK)
        ↓
OTLPSpanExporter → http://localhost:4318
        ↓
Fluent Bit (Docker) — receives, batches spans
        ↓
Langfuse — visualizes traces
```

## Project Structure

```
LLM Tracing/
├── app/
│   ├── config/
│   │   └── globals.py              # Environment variable loading
│   ├── core/
│   │   ├── logging.py              # Logger setup
│   │   └── singleton.py            # Singleton metaclass
│   └── tracing/
│       └── client_telemetry.py     # Core OTel tracing client
├── .env                            # Environment variables (never commit)
├── docker-compose.yml              # Runs Fluent Bit
├── fluent-bit.conf                 # Fluent Bit config (receives + forwards traces)
├── main.py                         # Entry point
├── pyproject.toml                  # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.11+
- Poetry
- Docker Desktop

## Setup

### 1. Clone and install dependencies

```bash
poetry install
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Tracing toggle
LANGFUSE_ENABLED=true

# Langfuse credentials (used by Fluent Bit only — not your app)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=your-langfuse-hostname       # no https://, no trailing slash

# LLM API key (OpenRouter example)
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Generate Fluent Bit auth token

Run this once to base64-encode your Langfuse credentials into `.env`:

```powershell
$public = (Get-Content .env | Where-Object { $_ -match "^LANGFUSE_PUBLIC_KEY=" }) -replace "^LANGFUSE_PUBLIC_KEY=", ""
$secret = (Get-Content .env | Where-Object { $_ -match "^LANGFUSE_SECRET_KEY=" }) -replace "^LANGFUSE_SECRET_KEY=", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$public`:$secret"))
$envContent = Get-Content .env | Where-Object { $_ -notmatch "^LANGFUSE_AUTH=" }
$envContent += "LANGFUSE_AUTH=$encoded"
$envContent | Set-Content .env
Write-Host "✅ LANGFUSE_AUTH saved to .env"
```

### 4. Start Fluent Bit

```bash
docker-compose up -d
```

Verify it's running:

```bash
docker-compose logs fluent-bit
```

You should see:
```
[input:opentelemetry:opentelemetry.0] listening on 0.0.0.0:4318
```

### 5. Run the app

```bash
poetry run python main.py
```

## How It Works

### `client_telemetry.py`

The core tracing client. It:
- Configures the OTel `TracerProvider`
- Sends spans to Fluent Bit via `OTLPSpanExporter` on `localhost:4318`
- Instruments the OpenAI Agents SDK automatically via `OpenAIAgentsInstrumentor`
- Exposes `start_span`, `start_observation`, and `start_generation` context managers for manual tracing

### `fluent-bit.conf`

Fluent Bit acts as the OTel collector:
- **INPUT**: Listens on port `4318` for OTLP traces from your app
- **OUTPUT**: Forwards traces to Langfuse with Basic auth

### `globals.py`

Loads environment variables from `.env` at import time.

### `singleton.py`

Ensures only one `TracingClient` instance exists per process, preventing duplicate spans.

## Usage Example

```python
from app.tracing.client_telemetry import TracingClient
from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

client = TracingClient()
client.instrument()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

agent = Agent(
    name="my-agent",
    instructions="You are a helpful assistant.",
    model=OpenAIChatCompletionsModel(
        model="google/gemini-2.0-flash-001",
        openai_client=openai_client,
    ),
)

with client.start_observation(name="my-workflow", input="Hello") as obs:
    obs.update_trace(user_id="user-123", session_id="session-abc")
    result = await Runner.run(agent, "Hello")
    obs.update(output=result.final_output)
```

## Viewing Traces

Open your Langfuse dashboard. Each run will appear as a trace with:
- Input and output
- Latency
- Token usage and cost
- User ID and session ID
- Tags and metadata
- Full span tree (agent → LLM calls → tool calls)

## Dependencies

| Package | Purpose |
|---|---|
| `opentelemetry-sdk` | Core OTel SDK |
| `opentelemetry-exporter-otlp` | OTLP HTTP exporter |
| `openinference-instrumentation-openai-agents` | Auto-instruments OpenAI Agents SDK |
| `openinference-instrumentation-openai` | Auto-instruments OpenAI client calls |
| `openai-agents` | OpenAI Agents SDK |
| `python-dotenv` | Loads `.env` file |

