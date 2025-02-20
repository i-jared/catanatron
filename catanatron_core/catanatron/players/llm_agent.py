from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from catanatron.models.enums import Action, ActionType
from catanatron.models.message import Message, MessageType
from dotenv import load_dotenv
import os
import logging
from langsmith import Client
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Updating game memory: {key}={value}")
        self.game_memory[key] = value
        
    def update_strategy(self, key: str, value: Any):
        logger.info(f"Updating strategy: {key}={value}")
        self.strategy_memory[key] = value

class CatanTools:
    """Collection of tools for the Catan agent"""
    def __init__(self, player_color: str, memory: CatanMemory):
        logger.info(f"Initializing CatanTools for player {player_color}")
        CatanTools._instance = self  # Store instance for static methods
        self.player_color = player_color
        self.memory = memory
        self.playable_actions = []  # Store available actions
        
        # Initialize base tools that are always available
        self.base_tools = [
            CatanTools.update_game_memory,
            CatanTools.update_strategy,
            # TODO: add send_message tool back in
            # CatanTools.send_message,
            CatanTools.select_action,
        ]
        logger.debug(f"Available tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in self.base_tools]}")
        
    @property
    def available_tools(self):
        """Returns list of currently available tools"""
        return self.base_tools
        
    @staticmethod
    @tool
    def update_game_memory(key: str, value: str) -> str:
        """Updates the game memory with new information. Use this to remember important game state."""
        logger.info(f"Tool call: update_game_memory(key={key}, value={value})")
        CatanTools._instance.memory.update_game_memory(key, value)
        return f"Updated game memory: {key}={value}"

    @staticmethod
    @tool
    def update_strategy(key: str, value: str) -> str:
        """Updates the long-term strategy memory. Use this to remember strategies that work well."""
        logger.info(f"Tool call: update_strategy(key={key}, value={value})")
        CatanTools._instance.memory.update_strategy(key, value)
        return f"Updated strategy: {key}={value}"

    @staticmethod
    @tool
    def send_message(content: str, to_color: Optional[str] = None) -> str:
        """Sends a message to other players. Leave to_color empty to send to all players."""
        logger.info(f"Tool call: send_message(content='{content}', to_color={to_color})")
        instance = CatanTools._instance
        instance.pending_message = Message(
            from_color=instance.player_color,
            to_color=to_color,
            message_type=MessageType.GENERAL,
            content={"text": content}
        )
        return f"Message queued: {content}"

    @staticmethod
    @tool
    def select_action(action_index: int) -> str:
        """Selects an action from the list of available actions by index"""
        logger.info(f"Tool call: select_action(action_index={action_index})")
        instance = CatanTools._instance
        try:
            index = int(action_index)
            if 0 <= index < len(instance.playable_actions):
                instance.pending_action = instance.playable_actions[index]
                logger.info(f"Selected action: {instance.pending_action}")
                return f"Selected action: {instance.pending_action}"
            else:
                error_msg = f"Invalid index. Please choose between 0 and {len(instance.playable_actions)-1}"
                logger.warning(error_msg)
                return error_msg
        except ValueError:
            error_msg = "Please provide a valid number"
            logger.warning(error_msg)
            return error_msg

    # Remove other action-specific tools since we're using select_action instead

class CatanAgent:
    """LLM-powered agent for playing Catan"""
    def __init__(self, player_color: str, provider: str = "anthropic"):
        logger.info(f"Initializing CatanAgent for player {player_color} using {provider}")
        self.player_color = player_color
        self.memory = CatanMemory()
        self.tools = CatanTools(player_color, self.memory)
        
        # Initialize LangSmith client for observability
        if os.getenv("LANGSMITH_API_KEY"):
            self.langsmith_client = Client()
            self.project_name = f"catan_agent_{player_color}_{uuid.uuid4().hex[:8]}"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            logger.info(f"LangSmith tracking enabled for project: {self.project_name}")
        else:
            logger.warning("LANGSMITH_API_KEY not found, observability disabled")
            self.langsmith_client = None

        self.memory_saver = MemorySaver()
        
        # Initialize LLM based on provider
        try:
            if provider == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
                logger.info("Initialized Anthropic Claude 3 Sonnet")
            
            elif provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
                logger.info("Initialized OpenAI GPT-4 Turbo")
            
            elif provider == "google":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY not found in environment")
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                logger.info("Initialized Google Gemini")
            
            elif provider == "mistral":
                if not os.getenv("MISTRAL_API_KEY"):
                    raise ValueError("MISTRAL_API_KEY not found in environment")
                self.llm = ChatMistralAI(model="mistral-large-latest")
                logger.info("Initialized Mistral Large")
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize {provider} LLM. Check your API key and try again.")
        
        # Create the agent with base tools
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools.available_tools,
            checkpointer=self.memory_saver
        )
        logger.info("Agent created successfully")

    async def decide_action(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> Action:
        """Decides the next action to take based on game state"""
        logger.info(f"Deciding action for {self.player_color}")
        logger.debug(f"Game state: {game_state}")
        logger.info(f"Available actions: {[f'{i}: {a.action_type} {a.value}' for i, a in enumerate(playable_actions)]}")
        
        if not playable_actions:
            logger.error("No playable actions available!")
            raise ValueError("No playable actions available")
        
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
        logger.debug(f"Run config: {config}")
        
        # Store playable actions in tools
        self.tools.playable_actions = playable_actions
        
        # Format the input for the agent
        prompt = self._format_game_state(game_state, playable_actions)
        logger.debug(f"Generated prompt: {prompt}")
        
        # Get agent's decision
        self.tools.pending_action = None
        try:
            logger.info("Starting agent decision process")
            chunks = []
            stream = self.agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            
            # Process the stream
            for chunk in stream:
                chunks.append(chunk)
                logger.debug(f"Agent stream chunk: {chunk}")
                
                # Process any generated messages
                if hasattr(self.tools, "pending_message"):
                    logger.info(f"Processing pending message: {self.tools.pending_message}")
                    del self.tools.pending_message
                    
                # Check if an action was decided
                if hasattr(self.tools, "pending_action"):
                    action = self.tools.pending_action
                    if action is None:
                        logger.warning("Agent selected None as action")
                        continue
                    logger.info(f"Action decided: {action}")
                    return action
            
            # Log the full conversation if no action was chosen
            logger.warning("No action chosen by agent. Full conversation:")
            for i, chunk in enumerate(chunks):
                logger.warning(f"Chunk {i}: {chunk}")
            
            # If no action was chosen, pick first available
            logger.warning("Defaulting to first available action")
            return playable_actions[0]
            
        except Exception as e:
            logger.error(f"Error during agent decision: {str(e)}", exc_info=True)
            logger.error("Full game state for debugging:")
            logger.error(f"Game state: {game_state}")
            logger.error(f"Playable actions: {playable_actions}")
            # Fallback to first action if there's an error
            logger.warning("Error occurred, falling back to first available action")
            return playable_actions[0]

    def _format_game_state(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> str:
        """Formats the game state and memory into a prompt for the agent"""
        logger.debug("Formatting game state into prompt")
        game_memory = self.memory.game_memory
        strategy_memory = self.memory.strategy_memory
        
        # Format available actions with indices
        action_list = "\n".join(
            f"{i}: {action.action_type} {action.value}" 
            for i, action in enumerate(playable_actions)
        )
        
        prompt = f"""
        You are playing Settlers of Catan. You need to choose your next action.
        
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
        - select_action: Choose an action from the numbered list above by providing the index number
        - update_game_memory: Store important information about the game state
        - update_strategy: Remember effective strategies for future reference
        - send_message: Communicate with other players
        
        First, analyze the situation and update your memory if needed.
        Then, you MUST select an action from the available options using the select_action tool with a valid index number.
        
        Remember:
        1. You MUST use select_action to choose your action
        2. The index must be a valid number from the list above
        3. You cannot skip making a decision
        """
        
        logger.debug(f"Generated prompt: {prompt}")
        return prompt