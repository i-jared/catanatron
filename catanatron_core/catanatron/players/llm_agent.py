from collections import deque
from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from catanatron.models.enums import Action, ActionType
from catanatron.models.message import Message, MessageType
from dotenv import load_dotenv
import os
import logging
from langsmith import Client
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import re
import time
from typing import Optional, Iterator, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
console = Console()

def log_prompt(prompt: str):
    """Log a shortened version of the prompt in a pretty format"""
    # Extract key sections
    sections = prompt.split("\n\n")
    
    # Find the sections we want
    game_state_start = None
    dev_cards_end = None
    actions_start = None
    
    for i, section in enumerate(sections):
        if section.startswith("Current Game State:"):
            game_state_start = i
        elif section.startswith("Your Development Cards:"):
            dev_cards_end = i + 1
        elif section.startswith("Available Actions:"):
            actions_start = i
            break
    
    if game_state_start is not None and dev_cards_end is not None and actions_start is not None:
        # Include game state through dev cards, plus available actions
        shortened = "\n\n".join(sections[game_state_start:dev_cards_end] + [sections[actions_start]])
    else:
        shortened = "Error: Could not find relevant sections"
    
    console.print(Panel(shortened, title="[blue]Prompt", border_style="blue"))

def log_response(response: str):
    """Log the agent's response in a pretty format"""
    # print(response)
    # Pattern to match content between content=' and ', with optional spaces
    pattern = r"content='([^']*?)'(?:,\s*additional_kwargs|\s*,|\s*\})"
    matches = re.findall(pattern, response)
    
    # Filter out empty matches and join with newlines
    content = "\n".join(match.strip() for match in matches if match.strip())
    
    # If we couldn't find any content, provide a default message
    if not content:
        content = "No explanation provided"
    
    console.print(Panel(content, title="[green]Response", border_style="green"))

def log_tool_call(tool_name: str, **kwargs):
    """Log a tool call in a pretty format"""
    args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    text = Text()
    text.append(tool_name, style="yellow")
    text.append("(")
    text.append(args_str)
    text.append(")")
    console.print(Panel(text, title="[yellow]Tool Call", border_style="yellow"))

# Try to load .env from package directory first, then from project root
package_env = Path(__file__).parent.parent.parent / '.env'
project_env = Path(__file__).parent.parent.parent.parent.parent / '.env'

if package_env.exists():
    load_dotenv(package_env)
elif project_env.exists():
    load_dotenv(project_env)
else:
    print("Warning: No .env file found in package or project root")

# Add debug print
print(f"ANTHROPIC_API_KEY loaded: {'ANTHROPIC_API_KEY' in os.environ}")

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
        log_tool_call("update_game_memory", key=key, value=value)
        instance = CatanTools._instance
        instance.memory.update_game_memory(key, value)
        # Track tool usage
        if hasattr(instance, 'agent'):
            instance.agent.current_decision_tools.append(("update_game_memory", f"{key}={value}"))
        return f"Updated game memory: {key}={value}"

    @staticmethod
    @tool
    def update_strategy(key: str, value: str) -> str:
        """Updates the long-term strategy memory. Use this to remember strategies that work well."""
        log_tool_call("update_strategy", key=key, value=value)
        instance = CatanTools._instance
        instance.memory.update_strategy(key, value)
        # Track tool usage
        if hasattr(instance, 'agent'):
            instance.agent.current_decision_tools.append(("update_strategy", f"{key}={value}"))
        return f"Updated strategy: {key}={value}"

    @staticmethod
    @tool
    def send_message(content: str, to_color: Optional[str] = None) -> str:
        """Sends a message to other players. Leave to_color empty to send to all players."""
        log_tool_call("send_message", content=content, to_color=to_color)
        instance = CatanTools._instance
        instance.pending_message = Message(
            from_color=instance.player_color,
            to_color=to_color,
            message_type=MessageType.GENERAL,
            content={"text": content}
        )
        # Track tool usage is handled in the agent's decide_action method
        return f"Message queued: {content}"

    @staticmethod
    @tool
    def select_action(action_index: int) -> str:
        """Selects an action from the list of available actions by index"""
        instance = CatanTools._instance
        try:
            index = int(action_index)
            if 0 <= index < len(instance.playable_actions):
                action = instance.playable_actions[index]
                instance.pending_action = action
                # Log with both index and action details
                log_tool_call("select_action", action_index=index, action=f"{action.action_type} {action.value}")
                # Track tool usage
                if hasattr(instance, 'agent'):
                    instance.agent.current_decision_tools.append(("select_action", f"index={action_index}, action={action}"))
                return f"Selected action: {action}"
            else:
                error_msg = f"Invalid index. Please choose between 0 and {len(instance.playable_actions)-1}"
                log_tool_call("select_action", action_index=index, error=error_msg)
                logger.warning(error_msg)
                instance.pending_action = None  # Explicitly set to None on invalid index
                return error_msg
        except ValueError:
            error_msg = "Please provide a valid number"
            log_tool_call("select_action", action_index=action_index, error=error_msg)
            logger.warning(error_msg)
            instance.pending_action = None  # Explicitly set to None on invalid number
            return error_msg

    # Remove other action-specific tools since we're using select_action instead

class CatanAgent:
    """LLM-powered agent for playing Catan"""
    def __init__(self, player_color: str, provider: str = "anthropic"):
        logger.info(f"Initializing CatanAgent for player {player_color} using {provider}")
        self.player_color = player_color
        self.memory = CatanMemory()
        self.tools = CatanTools(player_color, self.memory)
        self.tools.agent = self  # Connect tools to agent
        self.current_decision_tools = []  # Track tools used in current decision
        
        # Initialize LangSmith client for observability
        if os.getenv("LANGSMITH_API_KEY"):
            self.langsmith_client = Client()
            self.project_name = f"catan_agent_{player_color}_{uuid.uuid4().hex[:8]}"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            logger.info(f"LangSmith tracking enabled for project: {self.project_name}")
        else:
            logger.warning("LANGSMITH_API_KEY not found, observability disabled")
            self.langsmith_client = None

        self.memory_saver = LimitedMemorySaver(max_responses=10)
        
        # Initialize LLM based on provider
        try:
            if provider == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info("Initialized Anthropic Claude 3 Sonnet")
            
            elif provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("Initialized OpenAI GPT-4o-mini")
            
            elif provider == "google":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY not found in environment")
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
                logger.info("Initialized Google Gemini")
            
            elif provider == "xai":
                if not os.getenv("XAI_API_KEY"):
                    raise ValueError("XAI_API_KEY not found in environment")
                self.llm = ChatXAI(model="grok-beta", api_key=os.getenv("XAI_API_KEY"))
                logger.info("Initialized XAI model")
            
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
        """Decides what action to take based on the current game state"""
        if not playable_actions:
            logger.error("No playable actions available!")
            raise ValueError("No playable actions available")
        
        config = {
            "configurable": {
                # TODO: get the actual game state and valid num_turns
                "thread_id": f"{game_state.get('game_id', 'default')}-{game_state.get('num_turns', '0')}",
                "metadata": {
                    "player_color": self.player_color,
                    "turn_number": game_state.get("turn_number"),
                    "victory_points": game_state.get("victory_points"),
                }
            }
        }
        self.tools.playable_actions = playable_actions
        self.current_decision_tools = []  # Reset tool history for new decision
        prompt = self._format_game_state(game_state, playable_actions)
        log_prompt(prompt)
        
        try:
            chunks = []
            self.tools.pending_action = None
            valid_action_selected = False
            
            stream = self.agent.stream({"messages": [HumanMessage(content=prompt)]}, config=config)
            for chunk in stream:  # Sync iteration
                chunks.append(chunk)
                
                if hasattr(self.tools, "pending_message"):
                    log_response(f"Sending message: {self.tools.pending_message.content['text']}")
                    self.current_decision_tools.append(("send_message", self.tools.pending_message))
                    del self.tools.pending_message
                    # Update prompt with new tool history
                    prompt = self._format_game_state(game_state, playable_actions)
                    log_prompt(prompt)
                    stream = self.agent.stream({"messages": [HumanMessage(content=prompt)]}, config=config)
                    
                if hasattr(self.tools, "pending_action") and self.tools.pending_action is not None:
                    valid_action_selected = True
            
            # Log the agent's full response/reasoning
            full_response = "".join(str(chunk) for chunk in chunks)
            log_response(full_response)
            
            if valid_action_selected and self.tools.pending_action in playable_actions:
                return self.tools.pending_action
            
            logger.warning("No valid action chosen by agent, defaulting to first available action")
            return playable_actions[0]
        
        except Exception as e:
            logger.error(f"Error during agent decision: {str(e)}", exc_info=True)
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

        # Format resources in a readable way
        resources = game_state["resources"]
        resource_list = "\n".join(
            f"- {resource.title()}: {amount}" 
            for resource, amount in resources.items()
        )

        # Format development cards in a readable way
        dev_cards = game_state["development_cards"]
        dev_card_list = "\n".join(
            f"- {card.replace('_', ' ').title()}: {details['in_hand']} in hand, {details['played']} played"
            for card, details in dev_cards.items()
        )

        # Format recent actions by other players
        recent_actions = game_state.get("recent_actions", [])
        recent_actions_text = ""
        if recent_actions:
            recent_actions_list = "\n".join(
                f"- {action.color} player: {action.action_type} {action.value if action.value else ''}"
                for action in recent_actions
            )
            recent_actions_text = f"\nRecent Actions by Other Players:\n{recent_actions_list}\n"

        # Format buildings in a readable way
        buildings = game_state["buildings"]
        building_list = "\n".join([
            f"- Roads: {buildings['roads_available']} available",
            f"- Settlements: {buildings['settlements_available']} available",
            f"- Cities: {buildings['cities_available']} available",
            f"- Longest road length: {buildings['longest_road_length']}",
            f"- Has longest road: {buildings['has_longest_road']}",
            f"- Has largest army: {buildings['has_largest_army']}"
        ])

        # Format victory points in a readable way
        vp = game_state["victory_points"]
        vp_text = f"You have {vp['total']} victory points"

        # Building costs reference
        building_costs = """
Building Costs:
- Road: 1 Wood, 1 Brick
- Settlement: 1 Wood, 1 Brick, 1 Wheat, 1 Sheep
- City: 2 Wheat, 3 Ore
- Development Card: 1 Ore, 1 Wheat, 1 Sheep"""
        
        prompt = f"""
You are playing Settlers of Catan. You need to choose your next action.

Current Game State:
{vp_text}

{recent_actions_text}

Your Resources:
{resource_list}

Your Development Cards:
{dev_card_list}

Your Buildings:
{building_list}

{building_costs}

Game Memory:
{game_memory}

Strategy Memory:
{strategy_memory}

Available Actions:
{action_list}

What action should I take? Consider the game state, memory, and available actions carefully.
You have these tools available:
- select_action: Choose an action from the numbered list above by providing the index number
- update_game_memory: Store important information about the game that will help you.
- update_strategy: Remember effective strategies for future reference. persists across games.

First, analyze the situation and update your memory if needed. Second, reason through the best course of action.
Then, you MUST select an action from the available options using the select_action tool with a valid index number.

Remember:
1. You will use select_action to choose your action after analyzing the situation and updating your memory if needed.
2. The index must be a valid number from the list above - an exact match.
3. You cannot skip making a decision
"""
# todo: add back in to tool list
# - send_message: Communicate with other players
        
        logger.debug(f"Generated prompt: {prompt}")
        return prompt

def filter_messages(messages, max_ai_messages=10):
    """
    Filter the message history to include the last max_ai_messages AIMessages
    and their corresponding ToolMessages, ensuring no empty messages.

    Args:
        messages (list): List of messages in the conversation history.
        max_ai_messages (int): Maximum number of AIMessages to retain.

    Returns:
        list: Filtered list of messages.
    """
    def is_valid_content(content):
        """Helper to validate message content which can be string or list."""
        if content is None:
            return False
        if isinstance(content, str):
            return content.strip() != ''
        if isinstance(content, list):
            return len(content) > 0 and all(isinstance(item, str) and item.strip() != '' for item in content)
        return False

    # Filter out any messages with empty content
    valid_messages = [m for m in messages if (
        hasattr(m, 'content') and 
        is_valid_content(m.content)
    )]

    # Get the last max_ai_messages AIMessages
    ai_messages = [m for m in valid_messages if isinstance(m, AIMessage)][-max_ai_messages:]

    # Collect tool_call_ids from these AIMessages
    tool_call_ids = set()
    for ai_msg in ai_messages:
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                if isinstance(tool_call, dict) and 'id' in tool_call:
                    tool_call_ids.add(tool_call['id'])

    # Get ToolMessages that correspond to these tool_call_ids
    tool_messages = [
        m for m in valid_messages 
        if isinstance(m, ToolMessage) and 
        hasattr(m, 'tool_call_id') and 
        m.tool_call_id in tool_call_ids
    ]

    # Combine and preserve order
    filtered_messages = [
        m for m in valid_messages 
        if (isinstance(m, AIMessage) and m in ai_messages) or 
           (isinstance(m, ToolMessage) and m in tool_messages)
    ]

    return filtered_messages

class LimitedMemorySaver(MemorySaver):
    """
    A memory saver that limits the conversation history to the last max_ai_messages AIMessages
    and their corresponding ToolMessages, ensuring no empty messages.
    """
    def __init__(self, max_responses=10):
        super().__init__()
        self.max_ai_messages = max_responses
        self._response_queues = {}
        self._step_counters = {}

    def put(self, config: dict, checkpoint: dict, metadata: dict, new_versions: Any) -> dict:
        """
        Save the checkpoint, filtering the message history to include only valid messages.

        Args:
            config (dict): Configuration dictionary.
            checkpoint (dict): Checkpoint data containing channel_values.
            metadata (dict): Metadata dictionary.
            new_versions (Any): New versions data.

        Returns:
            dict: Updated configuration.
        """
        try:
            if "channel_values" in checkpoint and "messages" in checkpoint["channel_values"]:
                messages = checkpoint["channel_values"]["messages"]
                filtered_messages = filter_messages(messages, self.max_ai_messages)
                
                # Ensure we have at least one valid message
                if not filtered_messages:
                    logger.warning("No valid messages found after filtering")
                    return config
                    
                checkpoint["channel_values"]["messages"] = filtered_messages

                # Update response queue for this thread
                thread_id = config["configurable"]["thread_id"]
                if thread_id not in self._response_queues:
                    self._response_queues[thread_id] = deque(maxlen=self.max_ai_messages)
                self._response_queues[thread_id].extend(filtered_messages)

                # Update step counter
                if thread_id not in self._step_counters:
                    self._step_counters[thread_id] = 0
                self._step_counters[thread_id] += 1

            return super().put(config, checkpoint, metadata, new_versions)
        except Exception as e:
            logger.error(f"Error in put: {str(e)}", exc_info=True)
            return config

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the current conversation state."""
        try:
            thread_id = config["configurable"]["thread_id"]
            if thread_id not in self._response_queues or not self._response_queues[thread_id]:
                return None
            
            step = self._step_counters.get(thread_id, 0)
            messages = list(self._response_queues[thread_id])
            
            # Filter out any invalid messages using the same validation as filter_messages
            def is_valid_content(content):
                if content is None:
                    return False
                if isinstance(content, str):
                    return content.strip() != ''
                if isinstance(content, list):
                    return len(content) > 0 and all(isinstance(item, str) and item.strip() != '' for item in content)
                return False
                
            valid_messages = [
                m for m in messages 
                if hasattr(m, 'content') and is_valid_content(m.content)
            ]
            
            if not valid_messages:
                return None
                
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"messages": valid_messages}
            checkpoint["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00", time.gmtime())
            
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata={"step": step},
                parent_config=None,
            )
        except Exception as e:
            logger.error(f"Error in get_tuple: {str(e)}", exc_info=True)
            return None

    def list(self, config: dict, *, filter: Optional[dict] = None, before: Optional[dict] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        """List available checkpoints for the conversation."""
        try:
            checkpoint_tuple = self.get_tuple(config)
            if checkpoint_tuple is not None:
                yield checkpoint_tuple
        except Exception as e:
            logger.error(f"Error in list: {str(e)}", exc_info=True)