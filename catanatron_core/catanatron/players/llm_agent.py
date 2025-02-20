from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from google import genai
from langchain_mistralai import ChatMistralAI
from catanatron.models.enums import Action, ActionType
from catanatron.models.message import Message, MessageType
from dotenv import load_dotenv
import os
from langsmith import Client
from langgraph.checkpoint import BaseCheckpointer
import uuid

# Try to load .env from package directory first, then from project root
package_env = Path(__file__).parent.parent.parent / '.env'
project_env = Path(__file__).parent.parent.parent.parent.parent / '.env'

if package_env.exists():
    load_dotenv(package_env)
elif project_env.exists():
    load_dotenv(project_env)
else:
    print("Warning: No .env file found in package or project root")

class CatanMemory:
    """Stores game state and strategy information"""
    def __init__(self):
        self.game_memory: Dict[str, Any] = {}
        self.strategy_memory: Dict[str, Any] = {}
        
    def update_game_memory(self, key: str, value: Any):
        self.game_memory[key] = value
        
    def update_strategy(self, key: str, value: Any):
        self.strategy_memory[key] = value

class CatanTools:
    """Collection of tools for the Catan agent"""
    def __init__(self, player_color: str, memory: CatanMemory):
        self.player_color = player_color
        self.memory = memory
        self.playable_actions = []  # Store available actions
        
        # Initialize base tools that are always available
        self.base_tools = [
            self.update_game_memory,
            self.update_strategy,
            # TODO: add send_message tool back in
            # self.send_message,
            self.select_action,
        ]
        
    @property
    def available_tools(self):
        """Returns list of currently available tools"""
        return self.base_tools
        
    @tool
    def update_game_memory(self, key: str, value: str) -> str:
        """Updates the game memory with new information. Use this to remember important game state."""
        self.memory.update_game_memory(key, value)
        return f"Updated game memory: {key}={value}"

    @tool
    def update_strategy(self, key: str, value: str) -> str:
        """Updates the long-term strategy memory. Use this to remember strategies that work well."""
        self.memory.update_strategy(key, value)
        return f"Updated strategy: {key}={value}"

    @tool
    def send_message(self, content: str, to_color: Optional[str] = None) -> str:
        """Sends a message to other players. Leave to_color empty to send to all players."""
        self.pending_message = Message(
            from_color=self.player_color,
            to_color=to_color,
            message_type=MessageType.GENERAL,
            content={"text": content}
        )
        return f"Message queued: {content}"

    @tool
    def select_action(self, action_index: int) -> str:
        """Selects an action from the list of available actions by index"""
        try:
            index = int(action_index)
            if 0 <= index < len(self.playable_actions):
                self.pending_action = self.playable_actions[index]
                return f"Selected action: {self.pending_action}"
            else:
                return f"Invalid index. Please choose between 0 and {len(self.playable_actions)-1}"
        except ValueError:
            return "Please provide a valid number"

    # Remove other action-specific tools since we're using select_action instead

class CatanAgent:
    """LLM-powered agent for playing Catan"""
    def __init__(self, player_color: str, provider: str = "anthropic"):
        self.player_color = player_color
        self.memory = CatanMemory()
        self.tools = CatanTools(player_color, self.memory)
        
        # Initialize LangSmith client for observability
        if os.getenv("LANGSMITH_API_KEY"):
            self.langsmith_client = Client()
            self.project_name = f"catan_agent_{player_color}_{uuid.uuid4().hex[:8]}"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            print(f"LangSmith tracking enabled for project: {self.project_name}")
        else:
            print("Warning: LANGSMITH_API_KEY not found, observability disabled")
            self.langsmith_client = None

        # Initialize checkpointer with project info
        self.memory_saver = MemorySaver(
            project_name=self.project_name if self.langsmith_client else None,
            client=self.langsmith_client
        )
        
        # Initialize LLM based on provider
        try:
            if provider == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
            
            elif provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
            
            elif provider == "deepseek":
                if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_API_BASE"):
                    raise ValueError("DEEPSEEK_API_KEY or DEEPSEEK_API_BASE not found in environment")
                self.llm = ChatOpenAI(
                    model="deepseek-chat-33b",
                    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
                    max_tokens=4096
                )
            
            elif provider == "google":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY not found in environment")
                # Initialize Gemini
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                # Create a client
                model = genai.GenerativeModel('gemini-2.0-flash')
                # Wrap in LangChain format for compatibility
                self.llm = GeminiWrapper(model)
            
            elif provider == "mistral":
                if not os.getenv("MISTRAL_API_KEY"):
                    raise ValueError("MISTRAL_API_KEY not found in environment")
                self.llm = ChatMistralAI(model="mistral-large-latest")
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            # Log the full error but show a simplified message
            print(f"Error initializing LLM: {str(e)}")
            raise ValueError(f"Failed to initialize {provider} LLM. Check your API key and try again.")
        
        # Create the agent with base tools
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools.available_tools,
            checkpointer=self.memory_saver
        )

    async def decide_action(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> Action:
        """Decides the next action to take based on game state"""
        # Add run metadata for better tracking
        config = {
            "configurable": {
                "thread_id": str(game_state.get("game_id", "default")),
                "metadata": {
                    "player_color": self.player_color,
                    "turn_number": game_state.get("turn_number"),
                    "victory_points": game_state.get("victory_points"),
                }
            }
        }
        
        # Store playable actions in tools
        self.tools.playable_actions = playable_actions
        
        # Format the input for the agent
        prompt = self._format_game_state(game_state, playable_actions)
        
        # Get agent's decision
        self.tools.pending_action = None
        try:
            for chunk in self.agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            ):
                # Process any generated messages
                if hasattr(self.tools, "pending_message"):
                    # Handle message sending...
                    del self.tools.pending_message
                    
                # Check if an action was decided
                if hasattr(self.tools, "pending_action"):
                    return self.tools.pending_action
                    
            # If no action was chosen, pick first available
            return playable_actions[0]
            
        except Exception as e:
            print(f"Error during agent decision: {str(e)}")
            # Fallback to first action if there's an error
            return playable_actions[0]

    def _format_game_state(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> str:
        """Formats the game state and memory into a prompt for the agent"""
        game_memory = self.memory.game_memory
        strategy_memory = self.memory.strategy_memory
        
        # Format available actions with indices
        action_list = "\n".join(
            f"{i}: {action.action_type} {action.value}" 
            for i, action in enumerate(playable_actions)
        )
        
        return f"""
        Game State:
        {game_state}
        
        Game Memory:
        {game_memory}
        
        Strategy Memory:
        {strategy_memory}
        
        Available Actions:
        {action_list}
        
        What action should I take? Consider the game state, memory, and available actions carefully.
        You have these tools available:
        - select_action: Choose an action from the numbered list above
        - update_game_memory: Store important information about the game state
        - update_strategy: Remember effective strategies for future reference
        - send_message: Communicate with other players
        
        First, analyze the situation and update your memory if needed.
        Then, select an action from the available options using the select_action tool.
        """

class GeminiWrapper:
    """Wrapper class to make Gemini API compatible with LangChain interface"""
    def __init__(self, model):
        self.model = model
    
    async def ainvoke(self, messages, **kwargs):
        """Async invoke method for LangChain compatibility"""
        # Convert LangChain messages to Gemini format
        prompt = " ".join([msg.content for msg in messages])
        response = await self.model.generate_content_async(prompt)
        return HumanMessage(content=response.text)
    
    def invoke(self, messages, **kwargs):
        """Sync invoke method for LangChain compatibility"""
        prompt = " ".join([msg.content for msg in messages])
        response = self.model.generate_content(prompt)
        return HumanMessage(content=response.text) 