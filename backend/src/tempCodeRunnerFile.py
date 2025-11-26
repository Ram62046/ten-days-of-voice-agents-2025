import logging

# Standard Python Imports
import json  
import time  
import os    

from dotenv import load_dotenv

# LiveKit Agent Core Imports
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
# New/Updated LiveKit versions require LLM types from lk_types
from livekit.agents.lk_types import LLMResponse, LLMRequest
# LiveKit Plugin Imports
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ========================================================
# Day 2: Barista Logic
# ========================================================

# 1. INITIAL ORDER STATE
INITIAL_ORDER_STATE = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}

# 2. SYSTEM PROMPT (Barista Persona)
SYSTEM_PROMPT = """
You are 'Alex', a friendly and expert barista at 'The Murf Latte Lounge'. Your goal is to take a complete coffee order. You MUST fill all fields in the JSON object provided.

1.  **Analyze the current order state and the user's input.**
2.  **Update the 'order_state'** JSON object with any new information you find.
3.  **If the order is INCOMPLETE** (any required field is None), ask a single, clarifying question for the NEXT missing field.
4.  **If the order is COMPLETE** (all required fields are filled), confirm the final order and set the 'complete' status to True.

Your response MUST be a **valid JSON object** with the following three keys:
-   `order_state`: The complete, updated JSON state (must include all fields).
-   `response_text`: Your spoken response (the question or the confirmation).
-   `complete`: A boolean (True/False) indicating if the order is ready to be saved.
"""

# 3. JSON SAVE FUNCTION
def save_completed_order(order_data):
    """Saves the completed order to a unique JSON file."""
    if not os.path.exists("orders"):
        os.makedirs("orders")

    timestamp = int(time.time())
    filename = f"orders/completed_order_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(order_data, f, indent=4)
    
    print(f"\n--- ORDER COMPLETE! Saved to: {filename} ---")
    return filename


class Assistant(Agent):
    def __init__(self) -> None:
        
        # Day 2: Barista persona सेट करें
        super().__init__(instructions=SYSTEM_PROMPT)
        
        # Day 2: वर्तमान ऑर्डर स्टेट को ट्रैक करें
        self.current_order_state = INITIAL_ORDER_STATE.copy() 

    # Day 2: LLM के डिफ़ॉल्ट व्यवहार को ओवरराइड करने के लिए
    async def process_llm_request(self, req: LLMRequest) -> LLMResponse:
        """LLM को state और prompt भेजता है, और completion पर JSON को सेव करता है।"""
        
        # 1. LLM को भेजने के लिए इनपुट तैयार करें
        llm_input = f"""
        Current Order State: {json.dumps(self.current_order_state, indent=2)}
        User Input: "{req.text}"
        ---
        Based on the SYSTEM_PROMPT and the above, please provide the new state and your response text as a valid JSON object.
        """
        
        # 2. LLM से raw JSON response प्राप्त करें
        llm_response_text = ""
        llm_response_stream = req.llm.chat(
            history=req.history, 
            prompt=llm_input,
            system_prompt=SYSTEM_PROMPT, 
        )
        async for chunk in llm_response_stream:
            llm_response_text += chunk.text
        
        # 3. JSON को पार्स करें और State अपडेट करें
        try:
            # LLM अक्सर JSON को ```json ... ``` के अंदर भेजता है। इसे साफ़ करें।
            cleaned_text = llm_response_text.strip().replace("```json", "").replace("```", "").strip()
            response_data = json.loads(cleaned_text)
            
            new_order_state = response_data.get('order_state', self.current_order_state)
            response_text = response_data.get('response_text', "Sorry, I lost my place. Can you repeat?")
            is_complete = response_data.get('complete', False)
            
            # State Update
            self.current_order_state = new_order_state
            
            # 4. Check for Completion and Save
            if is_complete:
                save_completed_order(self.current_order_state)
                # is_final=True बातचीत को समाप्त करता है
                return LLMResponse(
                    text=response_text,
                    is_final=True 
                )
            
            return LLMResponse(
                text=response_text
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return LLMResponse(text="I'm sorry, I'm having trouble processing your order. Could you start over?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice AI pipeline setup
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
