# ======================================================
# 💼 DAY 5: AI SALES DEVELOPMENT REP (SDR)
# 🎯 COMPANY: LeadZen AI - B2B AI Sales Agent
# 🚀 FEATURES: FAQ Retrieval, Lead Qualification, JSON Database
# ======================================================

import logging
import json
import os
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass, asdict

# --- LiveKit Imports ---
from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ)
# ======================================================

FAQ_FILE = "leadzen_faq.json"
LEADS_FILE = "leadzen_leads_db.json"

# Default FAQ data for "LeadZen AI"
DEFAULT_FAQ = [
    {
        "question": "What does LeadZen AI actually do?",
        "answer": "LeadZen AI is an autonomous AI Sales Agent platform. It uses predictive analytics to find high-intent prospects, performs hyper-personalized multi-channel outreach (email, LinkedIn, voice), qualifies the leads, and only hands off warm, sales-qualified leads (SQLs) to your human team."
    },
    {
        "question": "Who is the ideal customer for LeadZen AI?",
        "answer": "Our platform is best suited for B2B SaaS, E-commerce, and high-growth technology startups that have a clear Ideal Customer Profile (ICP) and are looking to automate their top-of-funnel sales process to scale quickly."
    },
    {
        "question": "What is LeadZen AI's pricing model?",
        "answer": "Our primary model is based on **pay-per-qualified-lead**, meaning you only pay for the sales-qualified opportunities we deliver, minimizing upfront risk. We have a Starter plan and a Growth plan. Specific monthly costs are customized based on lead targets."
    },
    {
        "question": "Do you offer a free trial or a free tier?",
        "answer": "We do not offer a perpetually free tier, but we offer a **14-day proof-of-concept (POC) pilot**. This allows you to test the quality of the leads we generate before committing to a long-term plan. This pilot is usually heavily discounted or free, depending on the scope of work."
    },
    {
        "question": "How long does it take to get set up?",
        "answer": "The initial setup and AI training phase typically takes **7 to 10 days**. This includes integrating with your CRM, defining your ICP, and training our AI agent on your product messaging."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            print(f"📄 Creating default FAQ file at: {path}")
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            # Return as a string for the LLM to use in the System Prompt
            return json.dumps(json.load(f))
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

# Load the FAQ into a global variable for the agent's instructions
STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None
    
    def to_summary(self):
        return {
            "name": self.name,
            "company": self.company,
            "role": self.role,
            "use_case": self.use_case,
            "timeline": self.timeline,
        }

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's full name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's professional email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title or role, e.g., Head of Sales")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to achieve with LeadZen AI, e.g., 'Find SaaS leads'")] = None,
    team_size: Annotated[Optional[str], Field(description="Number of people in their sales/SDR team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start, e.g., 'Now', 'next quarter', 'exploring'")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation.
    Call this immediately when the user provides any of these pieces of information.
    """
    profile = ctx.userdata.lead_profile
    
    # Update only fields that are provided (not None)
    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline
    
    # Simple console log to track progress
    print(f"📝 UPDATING LEAD: {profile.to_summary()}")
    return "Lead profile updated in the system. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the complete lead to the database and signals the end of the call.
    Call this when the user says goodbye, 'that's all', or 'I'm done'.
    """
    profile = ctx.userdata.lead_profile
    
    # Save to JSON file (Append mode)
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
    
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()
    
    # Read existing, append, write back (Simple JSON DB)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding='utf-8') as f:
                existing_data = json.load(f)
        except: pass
    
    existing_data.append(entry)
    
    with open(db_path, "w", encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    
    # Generate the final verbal summary for the agent
    name = profile.name or "there"
    email = profile.email or "the email you provided"
    use_case = profile.use_case or "your general inquiry"
    
    summary = (
        f"Thank you, {name}! I've successfully logged your information regarding "
        f"{use_case}. A specialist will review your details and email you at "
        f"{email} shortly. Goodbye!"
    )
    
    return summary

# ======================================================
# 🧠 4. AGENT DEFINITION
# ======================================================

class LeadZenSDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Anjali', a friendly, professional, and highly efficient Sales Development Rep (SDR) for **LeadZen AI**.
            Your goal is to qualify the user and answer their questions using the FAQ.

            📘 **YOUR KNOWLEDGE BASE (FAQ):**
            {STORE_FAQ_TEXT}
            
            🎯 **YOUR GOAL:**
            1. **Qualify the Lead:** Naturally guide the conversation to collect Name, Company, Role, Email, Use Case, Team Size, and Timeline.
            2. **Answer Questions:** Use the FAQ content ONLY to answer product, service, or pricing questions.
            
            ⚙️ **BEHAVIOR:**
            - **Be Conversational:** Answer a user's question first, then use a polite transition to ask for the next piece of lead detail. Example: "Our pilot program is 14 days, which is great for testing lead quality. To send you the details, what's the best email for you?"
            - **Capture Data:** Use `update_lead_profile` immediately when you hear any new piece of lead information.
            - **Closing:** When the user indicates the call is over (e.g., "Thanks, that's all", "I'm done"), call `submit_lead_and_end`.

            🚫 **RESTRICTIONS:**
            - **Do NOT** make up prices or details not explicitly found in the FAQ. If you don't know the answer, say "I'll check with our product team and email you the specific detail."
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 5. ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    """Pre-load models before the job starts."""
    # Ensure VAD model is loaded only once
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💼" * 25)
    print(f"🚀 STARTING SDR SESSION for LeadZen AI in room: {ctx.room.name}")
    
    # 1. Initialize State (The lead profile for this session)
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent Pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie", 
            style="Promo",       
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # 3. Start the Session
    await session.start(
        agent=LeadZenSDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # Optional: Enable Noise Cancellation for cleaner audio
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
