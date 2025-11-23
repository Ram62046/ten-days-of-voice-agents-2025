import logging
import json
import os
from datetime import datetime
from typing import Annotated, Literal, List
from dataclasses import dataclass, field
from pydantic import Field

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🛒 ORDER MANAGEMENT SYSTEM
# ======================================================

@dataclass
class OrderState:
    """Coffee shop order state"""
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: List[str] = field(default_factory=list)
    name: str | None = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        # Checks if all fields (except optional extras list) are non-None and not empty string
        return all([
            self.drinkType is not None,
            self.size is not None,
            self.milk is not None,
            self.name is not None
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name
        }
    
    def get_summary(self) -> str:
        """Get friendly order summary for logging"""
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        if not self.is_complete():
            return "Order in progress..."
        
        return f"{self.size.upper()} {self.drinkType.title()} with {self.milk.title()} milk{extras_text} for {self.name}"

@dataclass
class Userdata:
    """User session data attached to RunContext"""
    order: OrderState
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 💾 ORDER STORAGE & PERSISTENCE
# ======================================================
def get_orders_folder():
    """Get the orders directory path (relative to backend/)"""
    base_dir = os.path.dirname(__file__)    # src/
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    folder = os.path.join(backend_dir, "orders")
    os.makedirs(folder, exist_ok=True)
    return folder

def save_order_to_json(order: OrderState) -> str:
    """Save order to JSON file"""
    folder = get_orders_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{timestamp}.json"
    path = os.path.join(folder, filename)

    try:
        order_data = order.to_dict()
        order_data["timestamp"] = datetime.now().isoformat()
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(order_data, f, indent=4, ensure_ascii=False)
        
        print("\n" + "✅" * 30)
        print("🎉 ORDER SAVED SUCCESSFULLY!")
        print(f"📁 Location: {path}")
        print("✅" * 30 + "\n")
        
        return path
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR SAVING ORDER: {e}")
        raise e

# ======================================================
# 🛠️ BARISTA AGENT FUNCTION TOOLS
# ======================================================

@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "coffee", "cold brew", "matcha"],
        Field(description="The type of coffee drink the customer wants"),
    ],
) -> str:
    """Set the drink type. Call when customer specifies which coffee they want."""
    ctx.userdata.order.drinkType = drink
    print(f"✅ DRINK SET: {drink.upper()}")
    return f"Excellent choice! One {drink} coming up!"

@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large", "extra large"],
        Field(description="The size of the drink"),
    ],
) -> str:
    """Set the size. Call when customer specifies drink size."""
    ctx.userdata.order.size = size
    print(f"✅ SIZE SET: {size.upper()}")
    return f"{size.title()} size - perfect!"

@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="The type of milk for the drink"),
    ],
) -> str:
    """Set milk preference. Call when customer specifies milk type."""
    ctx.userdata.order.milk = milk
    print(f"✅ MILK SET: {milk.upper()}")
    return f"{milk.title()} milk - great choice!"

@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        List[Literal["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "honey"]] | None,
        Field(description="List of extras, or empty/None for no extras"),
    ] = None,
) -> str:
    """Set extras. Call when customer specifies add-ons or says no extras."""
    ctx.userdata.order.extras = extras if extras else []
    print(f"✅ EXTRAS SET: {ctx.userdata.order.extras}")
    if ctx.userdata.order.extras:
        return f"Added {', '.join(ctx.userdata.order.extras)} - making it special!"
    return "No extras - keeping it classic and delicious!"

@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="Customer's name for the order")],
) -> str:
    """Set customer name. Call when customer provides their name."""
    ctx.userdata.order.name = name.strip().title()
    print(f"✅ NAME SET: {ctx.userdata.order.name}")
    return f"Wonderful, {ctx.userdata.order.name}! Almost ready to complete your order!"

@function_tool
async def complete_order(ctx: RunContext[Userdata]) -> str:
    """Finalize and save order to JSON. ONLY call when ALL required fields are filled."""
    order = ctx.userdata.order
    
    if not order.is_complete():
        missing = []
        if not order.drinkType: missing.append("drink type")
        if not order.size: missing.append("size")
        if not order.milk: missing.append("milk")
        if not order.name: missing.append("customer name")
        
        print(f"❌ CANNOT COMPLETE - Missing: {', '.join(missing)}")
        return f"Almost there! Just need: {', '.join(missing)}"
    
    print(f"🎉 ORDER READY FOR COMPLETION: {order.get_summary()}")
    
    try:
        save_order_to_json(order)
        extras_text = f" with {', '.join(order.extras)}" if order.extras else ""
        
        return f"PERFECT! Your {order.size} {order.drinkType} with {order.milk} milk{extras_text} is confirmed, {order.name}! We're preparing your drink now."
        
    except Exception as e:
        print(f"❌ ORDER SAVE FAILED: {e}")
        return "Order recorded but there was a small issue saving. Don't worry, we'll make your drink right away!"

# ======================================================
# ☕ AGENT DEFINITION
# ======================================================

class BaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a FRIENDLY and PROFESSIONAL barista named Alex at "The Murf Latte Lounge".
            
            MISSION: Take a coffee order by systematically collecting: Drink Type, Size, Milk, Extras, and Customer Name.
            
            PROCESS: Greet the customer warmly. Use the provided tools to set each parameter based on user input. Only call the complete_order tool when all information is confirmed and filled.
            
            STYLE: Be warm, enthusiastic, and professional. Ask one question at a time. Confirm choices as you go.
            """,
            tools=[
                set_drink_type,
                set_size,
                set_milk,
                set_extras,
                set_name,
                complete_order,
            ],
        )

# ======================================================
# 🎬 AGENT SESSION MANAGEMENT & BOOTSTRAP
# ======================================================
def prewarm(proc: JobProcess):
    """Preload VAD model"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main agent entrypoint - handles customer sessions"""
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create user session data with empty order
    userdata = Userdata(order=create_empty_order())
    
    # Run a quick test to ensure JSON saving works before starting session
    try:
        print("🧪 RUNNING ORDER SAVING TEST...")
        test_order = OrderState("test_latte", "test_medium", "test_oat", ["test_sugar"], "TestName")
        save_order_to_json(test_order)
        print("✅ Order saving test successful!")
    except Exception as e:
        print(f"🚨 CRITICAL: Order saving test failed! Check file permissions. Error: {e}")

    # Create session with userdata
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,  # Pass userdata to session
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

def create_empty_order():
    """Create a fresh order state (used by entrypoint)"""
    return OrderState()

if __name__ == "__main__":
    print("\n" + "⚡" * 25)
    print("🎬 STARTING COFFEE SHOP AGENT...")
    print("⚡" * 25 + "\n")
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
