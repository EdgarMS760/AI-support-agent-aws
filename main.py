"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, sys, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict, Optional
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
# Values below are redacted for this public repo (the Gateway uses the NONE
# authorizer, so the URL alone would grant unauthenticated tool access).
# Real values were set here during development and testing — see the test
# logs in this repo for evidence of the working, deployed agent.
GATEWAY_URL = "<gateway_url>"
KB_ID       = "<kbid>"
REGION      = "us-east-1"
MEMORY_ID   = "<mem_id>"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces: Dict[str, str] = {}
    for strategy in strategies:
        # Newer SDKs return "namespaceTemplates"; older ones return "namespaces".
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        if templates:
            namespaces[strategy["type"]] = templates[0]
    return namespaces


def _plain_text(message: Optional[dict]) -> Optional[str]:
    """Return the joined text of a message, or None if it's a tool call/result."""
    if not message:
        return None
    content = message.get("content") or []
    if any(("toolResult" in block or "toolUse" in block) for block in content):
        return None
    texts = [block["text"] for block in content if "text" in block]
    return " ".join(texts) if texts else None


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        # The original, un-annotated user text for this turn. We stash it here
        # because retrieve_customer_context mutates the message's content in
        # place (prepending retrieved memories); save_support_interaction must
        # not persist that annotated version back into memory, or the noise
        # compounds with every turn and dilutes future extraction.
        self._last_user_query = None

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        user_query = _plain_text(last_message)
        if not user_query:
            return

        self._last_user_query = user_query

        context_parts = []
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed for namespace {namespace}: {e}")
                continue

            for memory in memories or []:
                text = ((memory or {}).get("content") or {}).get("text", "")
                if text:
                    context_parts.append(f"[{strategy_type}] {text}")

        if context_parts:
            memory_block = "\n".join(context_parts)
            new_text = f"Customer Context:\n{memory_block}\n\n{user_query}"
            last_message["content"] = [{"text": new_text}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        agent_response = None
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            text = _plain_text(message)
            if text:
                agent_response = text
                break

        # Use the original query stashed by retrieve_customer_context, not the
        # (possibly memory-annotated) text sitting in messages right now.
        customer_query = self._last_user_query

        if not (customer_query and agent_response):
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning(f"Failed to save interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID == "<kbid>":
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return f"No relevant information found in the knowledge base for: {query}"

        chunks = [
            r["content"]["text"]
            for r in results
            if r.get("content", {}).get("text")
        ]
        return "\n---\n".join(chunks) if chunks else "No relevant information found."

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return f"Knowledge base search failed: {e}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = {json.dumps(tier)}
order_total = {order_total}
product_category = {json.dumps(product_category)}

value_per_point = 0.01  # $0.01 redemption value per loyalty point

max_redeemable_value = order_total * 0.5
max_points_from_value = int(max_redeemable_value / value_per_point)
points_redeemed = min(loyalty_points, max_points_from_value)
points_redeemed = (points_redeemed // 500) * 500  # floor to the nearest 500
points_value = points_redeemed * value_per_point

subtotal_after_points = order_total - points_value
tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount_amount = round(subtotal_after_points * tier_discount_pct, 2)
final_total = round(subtotal_after_points - tier_discount_amount, 2)
total_savings = round(order_total - final_total, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_value": round(points_value, 2),
    "tier_discount_pct": tier_discount_pct,
    "tier_discount_amount": tier_discount_amount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke("executeCode", {
                "language": "python",
                "code": code,
                "clearContext": True,
            })
            for event in response.get("stream", []):
                result_event = event.get("result")
                if result_event:
                    for block in result_event.get("content", []):
                        if "text" in block:
                            return block["text"]
                    return json.dumps(result_event)

        return json.dumps({"error": "No result returned from code interpreter"})

    except Exception as e:
        logger.warning(f"Code interpreter unavailable, using fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        final_total = round(order_total * (1 - tier_discount_pct), 2)
        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "final_total": final_total,
            "remaining_points": loyalty_points,
            "note": "Fallback calculation — code interpreter unavailable, points not redeemed.",
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        result_text = None
        try:
            with gateway_client:
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)

                agent = Agent(
                    model=model,
                    tools=tools,
                    hooks=[memory_hook],
                    system_prompt=(
                        "You are a helpful, concise customer support assistant for an "
                        "e-commerce platform. You can track orders, process refunds, "
                        "answer product and policy questions using the knowledge base, "
                        "calculate loyalty discounts, and browse the web when needed. "
                        "Always use your tools to verify facts — order status, policies, "
                        "and calculations — rather than guessing. "
                        "\n\nRefund workflow (follow exactly, in order, every time — even "
                        "if you already recall the order details): "
                        "1) Call get_order with the order_id to fetch its current total. "
                        "2) Only then call initiate_refund, passing that exact total as "
                        "the amount argument. Never call initiate_refund with amount=0 "
                        "or omit the amount."
                    ),
                )

                response = agent(user_input)
                result_text = response.message["content"][0]["text"]
        except Exception as e:
            # The MCP transport can raise on teardown (closing the streamable
            # HTTP connection) even after the agent already produced a valid
            # response. Don't discard a real answer because cleanup failed.
            if result_text is not None:
                logger.warning(f"Non-fatal error during Gateway client cleanup: {e}")
            else:
                raise

        return result_text

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return f"I'm sorry, something went wrong while processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)
    # The browser tool's destructor makes an untimed AWS network call to close
    # its remote session; it can hang during interpreter shutdown after we
    # already have our answer. Flush explicitly (os._exit skips normal stdio
    # flushing) and exit immediately instead of waiting on it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
