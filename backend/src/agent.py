## 💻 Day 6: SBI Fraud Agent (Murf TTS Variant)
# ======================================================
# 🏦 DAY 6: BANK FRAUD ALERT AGENT (SQLite DB variant)
# 🛡️ "State Bank of India (SBI)" - Fraud Detection & Resolution (USING MURF TTS)
# ======================================================

import logging
import os
import sqlite3
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass

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

# Standard LiveKit plugins
# AZURE removed; MURF added here.
from livekit.plugins import silero, google, deepgram, noise_cancellation, murf
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 💾 1. DATABASE SETUP (SQLite)
# ======================================================

DB_FILE = "fraud_db.sqlite"

@dataclass
class FraudCase:
    userName: str
    securityIdentifier: str
    cardEnding: str
    transactionName: str
    transactionAmount: str
    transactionTime: str
    transactionSource: str
    case_status: str = "pending_review"
    notes: str = ""


def get_db_path():
    """Returns the absolute path to the SQLite database file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)


def get_conn():
    """Returns a new SQLite connection with row factory set to sqlite3.Row."""
    path = get_db_path()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def seed_database():
    """Create SQLite DB and insert sample rows if empty."""
    conn = get_conn()
    cur = conn.cursor()

    # Create table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT NOT NULL,
            securityIdentifier TEXT,
            cardEnding TEXT,
            transactionName TEXT,
            transactionAmount TEXT,
            transactionTime TEXT,
            transactionSource TEXT,
            case_status TEXT DEFAULT 'pending_review',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # Insert sample data if table is empty
    cur.execute("SELECT COUNT(1) FROM fraud_cases")
    if cur.fetchone()[0] == 0:
        sample_data = [
            # Case 1: John (to be confirmed safe)
            (
                "John", "12345", "4242",
                "ABC Industry", "$450.00", "2:30 AM EST", "alibaba.com",
                "pending_review", "Automated flag: High value e-commerce transaction."
            ),
            # Case 2: Sarah (to be confirmed fraud)
            (
                "Sarah", "99887", "1199",
                "Unknown Crypto Exchange", "$2,100.00", "4:15 AM PST", "online_transfer",
                "pending_review", "Automated flag: Unusual location/merchant type."
            )
        ]
        cur.executemany(
            """
            INSERT INTO fraud_cases (
                userName, securityIdentifier, cardEnding, transactionName,
                transactionAmount, transactionTime, transactionSource, case_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_data,
        )
        conn.commit()
        print(f"✅ SQLite DB seeded at {DB_FILE} with 2 fraud cases.")

    conn.close()


# Initialize DB on load
seed_database()

# ======================================================
# 🧠 2. STATE MANAGEMENT
# ======================================================

@dataclass
class Userdata:
    active_case: Optional[FraudCase] = None

# ======================================================
# 🛠️ 3. FRAUD AGENT TOOLS (SQLite-backed)
# ======================================================

@function_tool
async def lookup_customer(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="The first or last name the user provides")],
) -> str:
    """Lookup a customer's pending fraud case in SQLite DB by name."""
    logger.info(f"🔎 LOOKING UP: {name}")
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM fraud_cases WHERE LOWER(userName) = LOWER(?) AND case_status = 'pending_review' LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return "User not found or case already resolved. Please ask the user to confirm the name."

        record = dict(row)
        # Store the case details in the session state
        ctx.userdata.active_case = FraudCase(
            userName=record["userName"],
            securityIdentifier=record["securityIdentifier"],
            cardEnding=record["cardEnding"],
            transactionName=record["transactionName"],
            transactionAmount=record["transactionAmount"],
            transactionTime=record["transactionTime"],
            transactionSource=record["transactionSource"],
            case_status=record["case_status"],
            notes=record["notes"],
        )

        return (
            f"Record Found for {record['userName']}. Security ID needed for verification is {record['securityIdentifier']}. "
            f"The suspicious transaction is {record['transactionAmount']} at {record['transactionName']}. "
            f"You must now ask the user for their Security Identifier."
        )

    except Exception as e:
        logger.error(f"Database lookup error: {str(e)}")
        return f"Database error occurred during lookup."


@function_tool
async def resolve_fraud_case(
    ctx: RunContext[Userdata],
    status: Annotated[str, Field(description="confirmed_safe or confirmed_fraud")],
    notes: Annotated[str, Field(description="A brief summary of the user's decision and the outcome.")],
) -> str:
    """Updates the status of the active fraud case in the SQLite database."""

    if not ctx.userdata.active_case:
        return "Error: No active fraud case selected. Cannot resolve."

    case = ctx.userdata.active_case
    case.case_status = status
    case.notes = notes

    try:
        conn = get_conn()
        cur = conn.cursor()

        # Update the database record
        cur.execute(
            """
            UPDATE fraud_cases
            SET case_status = ?, notes = ?, updated_at = datetime('now')
            WHERE userName = ?
            """,
            (case.case_status, case.notes, case.userName),
        )
        conn.commit()

        # Retrieve the updated timestamp for confirmation
        cur.execute("SELECT updated_at FROM fraud_cases WHERE userName = ?", (case.userName,))
        updated_at = cur.fetchone()[0]
        conn.close()

        logger.info(f"✅ CASE UPDATED: {case.userName} -> {status} at {updated_at}")

        # Provide a final confirmation message based on the outcome
        if status == "confirmed_fraud":
            return (
                f"Fraud confirmed. I have immediately blocked your card ending {case.cardEnding} and a dispute has been opened. "
                f"A replacement card will be issued to your address on file. Thank you for your cooperation."
            )
        elif status == "confirmed_safe":
            return (
                f"Transaction marked as safe. The temporary hold has been lifted on your card ending {case.cardEnding}. "
                f"Thank you for helping us protect your account."
            )
        else:
            return "Case status updated successfully."

    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
        return f"Error saving final status to the database: {e}"

# ======================================================
# 🤖 4. AGENT DEFINITION
# ======================================================

class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are 'Ravi', a Fraud Detection Specialist at **State Bank of India (SBI)**.
            Your goal is to quickly and professionally verify a suspicious transaction.

            **Protocol Steps:**
            1. **Greeting & Name:** Introduce yourself, state the purpose, and ask the customer for their **first name** to look up the case.
            2. **Case Load:** Immediately call `lookup_customer(name)` with the user's name.
            3. **Verification:** Ask the user to state their **Security Identifier** (e.g., "12345"). You must compare this to the 'Security ID (Expected)' value provided by the tool call result.
                - **If correct:** Proceed.
                - **If incorrect:** State politely, "I'm sorry, I cannot verify your identity with that information. I must end this call for security." and hang up.
            4. **Transaction Details:** If verified, read out the full suspicious transaction details stored in the active case.
            5. **Confirmation:** Ask: "Did you make this transaction?" (Yes/No).
            6. **Resolution:**
                - **User says YES:** Call `resolve_fraud_case(status='confirmed_safe', notes='Customer verified transaction.')`
                - **User says NO:** Call `resolve_fraud_case(status='confirmed_fraud', notes='Customer denied transaction.')`
            7. **Close:** Read the tool's final confirmation message and end the call professionally.
            """,
            tools=[lookup_customer, resolve_fraud_case],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    """Load VAD model during worker startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent session."""
    logger.info("🚀 STARTING SBI FRAUD ALERT SESSION (SQLite)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus", 
            style="Conversational", 
            text_pacing=True
        ), # TTS now uses Murf, requiring only MURF_API_KEY
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
