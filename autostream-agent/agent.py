"""
AutoStream Conversational AI Agent
====================================
A Social-to-Lead agentic workflow built with LangGraph + Gemini Flash.
Handles intent classification, RAG-powered Q&A, and lead capture.
"""
import os
import json
import re
from typing import TypedDict, Annotated, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# MOCK TOOL: Lead Capture
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API function to capture a qualified lead."""
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# MODEL RESOLUTION
# ─────────────────────────────────────────────

def _normalize_model_name(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _pick_first_available(available_models: List[str], prefixes: List[str]) -> Optional[str]:
    for prefix in prefixes:
        for model in available_models:
            if model == prefix or model.startswith(prefix):
                return model
    return None


def resolve_gemini_model(api_key: str, requested_model: Optional[str]) -> str:
    genai.configure(api_key=api_key)
    available_models = [
        _normalize_model_name(model.name)
        for model in genai.list_models()
        if "generateContent" in model.supported_generation_methods
    ]

    if not available_models:
        raise ValueError("No Gemini models with generateContent are available for your API key.")

    if requested_model:
        normalized = _normalize_model_name(requested_model)
        if normalized in available_models:
            return normalized
        print(
            f"[Warning] GEMINI_MODEL '{requested_model}' is not available for your API key. "
            "Falling back to the first available Gemini model."
        )

    preferred = _pick_first_available(available_models, ["gemini-1.5-flash", "gemini-1.5-pro"])
    if preferred:
        return preferred

    fallback = _pick_first_available(
        available_models,
        [
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    )
    if fallback:
        print(
            "[Warning] Gemini 1.5 is not available for your API key. "
            f"Using '{fallback}' instead."
        )
        return fallback

    raise ValueError(
        "No compatible Gemini models are available for your API key. "
        f"Available models: {', '.join(available_models)}."
    )


# ─────────────────────────────────────────────
# KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

class KnowledgeBase:
    """Loads and retrieves information from the local JSON knowledge base."""

    def __init__(self, kb_path: str = "knowledge_base/autostream_kb.json"):
        with open(kb_path, "r") as f:
            self.data = json.load(f)

    def retrieve(self, query: str) -> str:
        """Simple keyword-based retrieval from the knowledge base."""
        query_lower = query.lower()
        results = []

        # Always include company overview
        company = self.data["company"]
        results.append(f"Company: {company['name']} — {company['description']}")

        # Pricing info
        if any(kw in query_lower for kw in ["price", "plan", "cost", "pricing", "basic", "pro", "subscription", "pay", "how much", "fee"]):
            basic = self.data["pricing"]["basic_plan"]
            pro = self.data["pricing"]["pro_plan"]
            results.append(
                f"\nBasic Plan ({basic['price']}): {', '.join(basic['features'])}. Best for: {basic['best_for']}."
            )
            results.append(
                f"\nPro Plan ({pro['price']}): {', '.join(pro['features'])}. Best for: {pro['best_for']}."
            )

        # Policy info
        if any(kw in query_lower for kw in ["refund", "cancel", "policy", "support", "trial", "money back"]):
            policies = self.data["policies"]
            results.append(f"\nRefund Policy: {policies['refund_policy']}")
            results.append(f"Support: Basic — {policies['support']['basic']}. Pro — {policies['support']['pro']}.")
            results.append(f"Cancellation: {policies['cancellation']}")
            results.append(f"Free Trial: {policies['free_trial']}")

        # FAQ
        for faq in self.data["faq"]:
            if any(word in query_lower for word in faq["question"].lower().split()):
                results.append(f"\nQ: {faq['question']}\nA: {faq['answer']}")

        return "\n".join(results) if results else "No specific information found."

    def get_full_context(self) -> str:
        """Returns the entire KB as a formatted string for the system prompt."""
        return json.dumps(self.data, indent=2)


# ─────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str                      # "greeting" | "inquiry" | "high_intent" | "collecting_lead"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    turn_count: int


# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────

class AutoStreamAgent:
    """
    LangGraph-based conversational agent for AutoStream.
    Uses Gemini Flash as the LLM backbone.
    """

    def __init__(self, api_key: str, kb_path: str = "knowledge_base/autostream_kb.json"):
        requested_model = os.getenv("GEMINI_MODEL")
        model = resolve_gemini_model(api_key, requested_model)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
        )
        self.kb = KnowledgeBase(kb_path)
        self.graph = self._build_graph()

    # ── Intent Classification ──────────────────

    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent from the latest message."""
        last_message = state["messages"][-1].content

        intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intent classifier for AutoStream, a SaaS video editing product.
Classify the user's latest message into EXACTLY one of these intents:
- "greeting": casual hello, how are you, general pleasantries
- "inquiry": asking about features, pricing, plans, policies, comparisons
- "high_intent": user wants to sign up, try, buy, get started, or shows strong purchase interest
- "collecting_lead": user is in the process of providing their name/email/platform (already identified as high intent)

Return ONLY the intent word, nothing else."""),
            HumanMessage(content=f"Current intent state: {state['intent']}\nUser message: {last_message}")
        ])

        response = self.llm.invoke(intent_prompt.format_messages())
        raw_intent = response.content.strip().lower().replace('"', '').replace("'", "")

        # Preserve collecting_lead state if already gathering info
        if state["intent"] == "collecting_lead" and not all([state["lead_name"], state["lead_email"], state["lead_platform"]]):
            new_intent = "collecting_lead"
        else:
            new_intent = raw_intent if raw_intent in ["greeting", "inquiry", "high_intent", "collecting_lead"] else "inquiry"

        return {**state, "intent": new_intent, "turn_count": state["turn_count"] + 1}

    # ── Extract Lead Info ──────────────────────

    def extract_lead_info(self, state: AgentState) -> AgentState:
        """Try to extract name, email, and platform from the latest message."""
        last_message = state["messages"][-1].content

        extract_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Extract lead information from the user's message if present.
Return a JSON object with keys: "name", "email", "platform".
Use null for any field not found in the message.
Return ONLY raw JSON, no markdown, no explanation.
Examples of platform: YouTube, Instagram, TikTok, Facebook, Twitter, LinkedIn"""),
            HumanMessage(content=last_message)
        ])

        try:
            response = self.llm.invoke(extract_prompt.format_messages())
            raw = response.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            extracted = json.loads(raw)

            return {
                **state,
                "lead_name": extracted.get("name") or state["lead_name"],
                "lead_email": extracted.get("email") or state["lead_email"],
                "lead_platform": extracted.get("platform") or state["lead_platform"],
            }
        except Exception:
            return state

    # ── Generate Response ──────────────────────

    def generate_response(self, state: AgentState) -> AgentState:
        """Generate the agent's response based on intent and state."""
        intent = state["intent"]
        kb_context = self.kb.retrieve(state["messages"][-1].content)

        # Build missing fields message for lead collection
        missing = []
        if not state["lead_name"]:
            missing.append("full name")
        if not state["lead_email"]:
            missing.append("email address")
        if not state["lead_platform"]:
            missing.append("creator platform (e.g., YouTube, Instagram)")

        system_prompt = f"""You are Aria, a friendly and professional sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

KNOWLEDGE BASE:
{kb_context}

CURRENT STATE:
- Intent: {intent}
- Lead Name collected: {state['lead_name'] or 'Not yet'}
- Lead Email collected: {state['lead_email'] or 'Not yet'}
- Lead Platform collected: {state['lead_platform'] or 'Not yet'}
- Missing info: {', '.join(missing) if missing else 'None — all collected!'}
- Lead already captured: {state['lead_captured']}

INSTRUCTIONS:
1. If intent is "greeting": respond warmly, introduce AutoStream briefly, invite questions.
2. If intent is "inquiry": use the knowledge base to answer accurately. Be concise and helpful.
3. If intent is "high_intent": express enthusiasm, confirm their interest, then begin collecting lead info. Ask for ONE missing field at a time.
4. If intent is "collecting_lead": acknowledge what they shared, then ask for the next missing field (only one at a time). Once all three are collected (name, email, platform), confirm the details and tell them you're registering them.
5. Keep responses conversational, warm, and brief (2-4 sentences max unless explaining pricing).
6. NEVER ask for information you already have.
7. If lead is already captured, thank them and offer further assistance."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        response = self.llm.invoke(prompt.format_messages(messages=state["messages"]))
        new_messages = state["messages"] + [AIMessage(content=response.content)]
        return {**state, "messages": new_messages}

    # ── Trigger Lead Capture Tool ──────────────

    def trigger_lead_capture(self, state: AgentState) -> AgentState:
        """Call mock_lead_capture when all info is collected."""
        if all([state["lead_name"], state["lead_email"], state["lead_platform"]]) and not state["lead_captured"]:
            result = mock_lead_capture(state["lead_name"], state["lead_email"], state["lead_platform"])

            confirmation_msg = (
                f"🎉 You're all set, {state['lead_name']}! I've registered your interest in AutoStream's Pro Plan. "
                f"Our team will reach out to your email **{state['lead_email']}** shortly with next steps. "
                f"We're excited to help you grow your {state['lead_platform']} channel!"
            )
            new_messages = state["messages"] + [AIMessage(content=confirmation_msg)]
            return {**state, "messages": new_messages, "lead_captured": True}

        return state

    # ── Routing Logic ──────────────────────────

    def route_after_intent(self, state: AgentState) -> str:
        """Decide which node to run after intent classification."""
        if state["intent"] in ("high_intent", "collecting_lead"):
            return "extract_lead"
        return "generate_response"

    def route_after_extract(self, state: AgentState) -> str:
        """After extraction, check if all info is ready to capture."""
        if all([state["lead_name"], state["lead_email"], state["lead_platform"]]) and not state["lead_captured"]:
            return "trigger_lead_capture"
        return "generate_response"

    # ── Build LangGraph ────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("extract_lead", self.extract_lead_info)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("trigger_lead_capture", self.trigger_lead_capture)

        graph.set_entry_point("classify_intent")

        graph.add_conditional_edges("classify_intent", self.route_after_intent, {
            "extract_lead": "extract_lead",
            "generate_response": "generate_response",
        })

        graph.add_conditional_edges("extract_lead", self.route_after_extract, {
            "trigger_lead_capture": "trigger_lead_capture",
            "generate_response": "generate_response",
        })

        graph.add_edge("trigger_lead_capture", END)
        graph.add_edge("generate_response", END)

        return graph.compile()

    # ── Public Interface ───────────────────────

    def chat(self, user_input: str, state: AgentState) -> tuple[str, AgentState]:
        """Process a single user turn and return (response_text, new_state)."""
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        new_state = self.graph.invoke(state)
        last_ai = next(
            (m.content for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)),
            "I'm sorry, I couldn't generate a response."
        )
        return last_ai, new_state


# ─────────────────────────────────────────────
# INITIAL STATE FACTORY
# ─────────────────────────────────────────────

def create_initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="greeting",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        turn_count=0,
    )


# ─────────────────────────────────────────────
# CLI RUNNER
# ─────────────────────────────────────────────

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("\n" + "="*60)
    print("  AutoStream AI Agent — Powered by Gemini Flash + LangGraph")
    print("="*60)
    print("  Type 'exit' or 'quit' to end the conversation.\n")

    agent = AutoStreamAgent(api_key=api_key)
    state = create_initial_state()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nAgent: Thanks for chatting! Have a great day. 👋\n")
            break

        response, state = agent.chat(user_input, state)
        print(f"\nAria (AutoStream): {response}\n")

        if state["lead_captured"]:
            print("[System: Lead captured. Conversation can continue for support queries.]\n")


if __name__ == "__main__":
    main()
