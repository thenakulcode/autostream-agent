# AutoStream AI Agent

> **Social-to-Lead Agentic Workflow** — ServiceHive × Inflx Internship Assignment

A production-grade conversational AI agent that qualifies social media users into business leads for **AutoStream**, a fictional SaaS video editing platform for content creators. Built with LangGraph, Gemini 1.5 when available (with safe fallback), and a local RAG knowledge base.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Example Conversation](#example-conversation)
- [WhatsApp Deployment](#whatsapp-deployment)
- [Evaluation Criteria](#evaluation-criteria)

---

## Overview

Unlike a simple chatbot, this agent reasons over user intent, retrieves accurate product information from a local knowledge base, and executes a lead capture tool at precisely the right moment in the conversation — mirroring how a real sales agent would qualify a prospect.

The agent is named **Aria** and operates as an AutoStream sales assistant across a multi-turn conversation flow.

---

## Features

| Capability | Description |
|---|---|
| **Intent Classification** | Classifies every user message as `greeting`, `inquiry`, `high_intent`, or `collecting_lead` |
| **RAG-Powered Q&A** | Retrieves pricing, plan details, and policies from a local JSON knowledge base |
| **Lead Qualification** | Collects name, email, and creator platform one field at a time |
| **Tool Execution** | Calls `mock_lead_capture()` exactly once, only after all three fields are confirmed |
| **Memory** | Retains full conversation state across 6+ turns using LangGraph's typed state |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Agent Framework | LangGraph (LangChain) |
| LLM | Gemini 1.5 Flash/Pro (Google AI, auto-selected; falls back to latest Flash/Pro if unavailable) |
| Knowledge Base | Local JSON + keyword-based retrieval |
| State Management | LangGraph `AgentState` (TypedDict) |
| Lead Capture | `mock_lead_capture()` mock API function |

---

## Project Structure

```
autostream-agent/
│
├── agent.py                    # Core agent — LangGraph graph, nodes, routing logic
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env                        # Environment variables (not committed)
├── .env.example                # Template for environment setup
│
└── knowledge_base/
    └── autostream_kb.json      # RAG source — pricing, policies, FAQs
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Gemini API key — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Open `.env` and set:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional: set a specific Gemini 1.5 model from your account
# GEMINI_MODEL=gemini-1.5-flash-001
```

If `GEMINI_MODEL` is not set, the agent selects the first available Gemini 1.5 Flash/Pro model that supports `generateContent`. If Gemini 1.5 is not available for your API key, it falls back to the latest Flash/Pro model and prints a warning.

### 5. Run the Agent

```bash
python agent.py
```

A CLI chat interface will launch. Type naturally — the agent handles the rest.

---

## Architecture

This agent is built on **LangGraph**, a graph-based orchestration framework from LangChain, chosen specifically because it models the agent as a directed acyclic graph with explicit, inspectable state transitions — rather than opaque chain-of-thought loops. This makes the flow easy to debug, extend, and deploy in production.

### Graph Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ classify_intent │  ← Labels message: greeting / inquiry / high_intent / collecting_lead
└────────┬────────┘
         │
    ┌────┴─────────────────┐
    │                      │
    ▼                      ▼
┌──────────────┐    ┌──────────────────┐
│ extract_lead │    │ generate_response│
│  (if intent  │    │  (greeting or    │
│  is high)    │    │   inquiry)       │
└──────┬───────┘    └──────────────────┘
       │
  ┌────┴──────────────────────┐
  │                           │
  ▼                           ▼
┌──────────────────────┐  ┌──────────────────┐
│ trigger_lead_capture │  │ generate_response │
│ (all 3 fields ready) │  │ (ask next field)  │
└──────────────────────┘  └──────────────────┘
```

### State Management

A typed `AgentState` dictionary is passed through every node in the graph, preserving full context across all conversation turns:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]   # Full conversation history
    intent: str                   # Current classified intent
    lead_name: Optional[str]      # Collected lead fields
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool           # Ensures tool fires exactly once
    turn_count: int
```

### RAG Pipeline

The `KnowledgeBase` class loads `autostream_kb.json` at startup and uses keyword-based retrieval to inject only contextually relevant data into the system prompt per turn — keeping token usage lean and responses accurate.

---

## Example Conversation

```
You        → Hi there!
Aria       → Hey! Welcome to AutoStream 👋 I'm Aria, your AutoStream assistant.
              How can I help you today?

You        → Tell me about your pricing plans
Aria       → We have two plans:
              • Basic — $29/month: 10 videos, 720p, email support
              • Pro   — $79/month: Unlimited videos, 4K, AI captions, 24/7 support
              Which one sounds right for you?

You        → The Pro plan sounds great, I want to sign up for my YouTube channel
Aria       → Amazing choice! I'd love to get you set up. Could I start with your full name?

You        → My name is Nakul Sharma
Aria       → Great, Nakul! What's your email address?

You        → nakul@gmail.com
Aria       → Perfect! And which platform are you primarily creating for?

You        → YouTube

──────────────────────────────────────────
✅ Lead captured successfully!
   Name     : Nakul Sharma
   Email    : nakul@gmail.com
   Platform : YouTube
──────────────────────────────────────────

Aria       → 🎉 You're all set, Nakul! Our team will reach out to nakul@gmail.com
              with your Pro Plan details. Excited to help your YouTube channel grow!
```

---

## WhatsApp Deployment

To deploy this agent on WhatsApp, use the **WhatsApp Business Cloud API** from Meta.

### High-Level Architecture

```
User (WhatsApp)  →  Meta Webhook  →  FastAPI Server  →  LangGraph Agent  →  Response
```

### Integration Steps

**1. Register on Meta for Developers**
Create a WhatsApp Business app at [developers.facebook.com](https://developers.facebook.com) to obtain a phone number ID and access token.

**2. Build a FastAPI Webhook Server**

```python
from fastapi import FastAPI, Request
app = FastAPI()

sessions: dict[str, AgentState] = {}

@app.get("/webhook")
def verify(hub_challenge: str):
    return int(hub_challenge)   # Meta verification handshake

@app.post("/webhook")
async def receive_message(request: Request):
    payload = await request.json()
    phone = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text  = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    if phone not in sessions:
        sessions[phone] = create_initial_state()

    response, sessions[phone] = agent.chat(text, sessions[phone])
    send_whatsapp_message(phone, response)   # calls Meta POST /messages API
```

**3. Send Replies via Meta API**

Use `POST https://graph.facebook.com/v19.0/{phone_number_id}/messages` with your Bearer token to send outbound messages.

**4. Deploy & Register Webhook**

Deploy to any public HTTPS server (Railway, Render, AWS, etc.) and register your `/webhook` URL in the Meta Developer Console.

**5. Scale with Persistent State**

Replace the in-memory `sessions` dict with **Redis** or **PostgreSQL**, keyed by phone number, to support concurrent users and survive server restarts.

---

## Evaluation Criteria

| Criterion | Implementation |
|---|---|
| Agent Reasoning & Intent Detection | 4-class Gemini-based classifier with conditional LangGraph routing |
| RAG Knowledge Retrieval | Keyword retrieval from local JSON, injected per-turn into system prompt |
| State Management | Typed `AgentState` persisted across all turns via LangGraph |
| Tool Calling Logic | `mock_lead_capture()` fires once, only after all 3 fields are confirmed |
| Code Clarity & Structure | Modular classes, typed state, single-responsibility nodes |
| Real-World Deployability | WhatsApp webhook architecture documented; stateless graph scales horizontally |

---

## License

This project was built as part of a machine learning internship assignment for **ServiceHive (Inflx)**. For educational use only.
