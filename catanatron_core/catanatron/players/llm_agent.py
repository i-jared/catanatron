import os
import logging
import re
import uuid

from catanatron.models.enums import Action
from catanatron.models.message import Message, MessageType
from collections import deque
from dotenv import load_dotenv
from langsmith import Client
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Dict, List, Optional, Any
from typing import Optional, Dict, Any
from catanatron.players.chat import create_chat_provider

# *************************************
# ************* LOGGING ***************
# *************************************
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
    console.print(Panel(prompt, title="[blue]Prompt", border_style="blue"))

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


# *************************************
# *************** ENV *****************
# *************************************

package_env = Path(__file__).parent.parent.parent / '.env'
load_dotenv(package_env)


# *************************************
# ************* HELPERS ***************
# *************************************
class Message:
    """messages"""
    def __init__(self, reasoning: str, tool_call: str, tool_args, result: str):
        self.reasoning = reasoning
        self.tool_call= tool_call
        self.tool_args = tool_args
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to dictionary representation"""
        return {
            "reasoning": self.reasoning,
            "tool_call": self.tool_call,
            "tool_args": self.tool_args,
            "result": self.result
        }

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

class DecisionHistory:
    """Stores decision history for the agent"""
    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)
        
    def add_decision(self, reasoning: str, tool_name: str, tool_args: Dict[str, Any], result: str):
        """Add a tool call to the conversation history"""
        self.messages.append(Message(reasoning, tool_name, tool_args, result));

    def get_decisions(self):
        """Get all messages in the conversation history"""
        return list(self.messages)
        
    def clear(self):
        """Clear the conversation history"""
        self.messages.clear()

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
        logger.debug(f"Available tools: {[tool.__name__ for tool in self.base_tools]}")
        
    @property
    def available_tools(self):
        """Returns list of currently available tools"""
        return self.base_tools
        
    @staticmethod
    def update_game_memory(key: str, value: str) -> str:
        """Updates the game memory with new information. Use this to remember important game state."""
        log_tool_call("update_game_memory", key=key, value=value)
        instance = CatanTools._instance
        instance.memory.update_game_memory(key, value)
        # Track tool usage
        return f"Updated game memory: {key}={value}"

    @staticmethod
    def update_strategy(key: str, value: str) -> str:
        """Updates the long-term strategy memory. Use this to remember strategies that work well."""
        log_tool_call("update_strategy", key=key, value=value)
        instance = CatanTools._instance
        instance.memory.update_strategy(key, value)
        # Track tool usage
        return f"Updated strategy: {key}={value}"

    @staticmethod
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


# *************************************
# ************** AGENT ****************
# *************************************
class LLMAgent:
    """LLM-powered agent for playing Catan"""
    def __init__(self, player_color: str, provider: str = "anthropic"):
        logger.info(f"Initializing CatanAgent for player {player_color} using {provider}")
        self.player_color = player_color
        self.memory = CatanMemory()
        self.tools = CatanTools(player_color, self.memory)
        self.decision_history = DecisionHistory(max_messages=10)
        
        # Initialize LangSmith client for observability
        if os.getenv("LANGSMITH_API_KEY"):
            self.langsmith_client = Client()
            self.project_name = f"catan_agent_{player_color}_{uuid.uuid4().hex[:8]}"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            logger.info(f"LangSmith tracking enabled for project: {self.project_name}")
        else:
            logger.warning("LANGSMITH_API_KEY not found, observability disabled")
            self.langsmith_client = None
        
        # Initialize chat provider
        try:
            self.llm = create_chat_provider(provider)
            logger.info(f"Initialized {provider} chat provider")
        except Exception as e:
            logger.error(f"Error initializing chat provider: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize {provider} chat provider. Check your API key and try again.")
        
        # Create a dictionary of available tools for the agent
        self.available_tools = {tool.__name__: tool for tool in self.tools.available_tools}
        logger.info("Agent created successfully")

    async def decide_action(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> Action:
        """Decides what action to take based on the current game state"""
        if not playable_actions:
            logger.error("No playable actions available!")
            raise ValueError("No playable actions available")
        
        # Set up metadata for tracking
        metadata = {
            "player_color": self.player_color,
            "turn_number": game_state.get("turn_number"),
            "victory_points": game_state.get("victory_points"),
            "game_id": game_state.get("game_id", "default"),
            "num_turns": game_state.get("num_turns", "0")
        }
        
        self.tools.playable_actions = playable_actions
        prompt = self._format_game_state(game_state, playable_actions)
        log_prompt(prompt)
        
        try:
            self.tools.pending_action = None
            valid_action_selected = False
            
            # Run the ReAct loop
            max_iterations = 10
            for i in range(max_iterations):
                if valid_action_selected:
                    break
                
                # Get and parse the LLM response
                response_data = await self._get_llm_response(prompt)
                
                # Parse the response for tool calls
                tool_call = self._parse_tool_calls(response_data)
                
                if tool_call:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    if tool_name in self.available_tools:
                        tool = self.available_tools[tool_name]
                        result = tool(**tool_args)
                        
                        # Add the tool call to conversation history
                        self.decision_history.add_decision(response_data.get("reasoning", ""), tool_name, tool_args, result)
                        
                        # Check if an action was selected
                        if tool_name == "select_action" and self.tools.pending_action is not None:
                            valid_action_selected = True
                            break
                    else:
                        logger.warning(f"Unknown tool: {tool_name}")
            
            if valid_action_selected and self.tools.pending_action in playable_actions:
                return self.tools.pending_action
            
            logger.warning("No valid action chosen by agent, defaulting to first available action")
            return playable_actions[0]
        
        except Exception as e:
            logger.error(f"Error during agent decision: {str(e)}", exc_info=True)
            logger.warning("Error occurred, falling back to first available action")
            return playable_actions[0]

    async def _get_llm_response(self, prompt: str) -> Dict[str, Any]:
        """Get a response from the LLM and parse it into a structured format"""
        try:
            response = await self.llm.chat(prompt)
            
            # Parse the JSON response
            json_pattern = r'```json\s*(.*?)\s*```'
            json_blocks = re.findall(json_pattern, str(response), re.DOTALL)
            
            if not json_blocks:
                # Try alternative format without json tag
                json_pattern = r'```\s*(\{.*?\})\s*```'
                json_blocks = re.findall(json_pattern, str(response), re.DOTALL)
            
            if not json_blocks:
                # Try to find JSON objects directly in the text
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                json_blocks = re.findall(json_pattern, str(response))
                
            if json_blocks:
                try:
                    import json
                    data = json.loads(json_blocks[0])
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
            return None

    def _parse_tool_calls(self, response_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse the response data into a tool call format"""
        if not response_data:
            return None
        
        # Extract thought and tool information
        thought = response_data.get("thought")
        tool = response_data.get("tool")
        tool_params = response_data.get("tool_params", {})
        
        if thought and tool:
            # Log the thought
            console.print(Panel(thought, title="[cyan]Thought", border_style="cyan"))
            
            # Return tool call info
            return {
                "name": tool,
                "args": tool_params
            }
        return None

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
        recent_actions_text = "N/A"
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

        decision_history = "\n".join([msg.to_dict().get("result", "") for msg in self.decision_history.get_decisions()])

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
        
        # Add ReAct format instructions
        react_instructions = """
You are an agent playing Settlers of Catan who thinks step by step to solve problems. Reason which tool is best for your situation, then choose a tool. 
You must use a tool. Follow this json response format:

```json
{
  "thought": <Consider the current situation and what to do next>
  "tool": <tool_name>,
  "tool_params": {
    <param1>: <value1>,
    <param2>: <value2>,
  }
}
```

Available tools:
- update_game_memory: Store important information about the game
  Parameters: {"key": "string", "value": "string"}
- update_strategy: Remember effective strategies for future reference
  Parameters: {"key": "string", "value": "string"}
- select_action: Choose an action from the numbered list by index
  Parameters: {"action_index": number}
"""
        
        prompt = f"""
{react_instructions}

Available Actions:
{action_list}



CURRENT GAME STATE:

{vp_text}

Your Resources: 
{resource_list}

Your Development Cards:
{dev_card_list}

Building Info:
{building_list}


{building_costs}


MEMORY:

Your Recent actions:
{decision_history}

Recent Actions by All Players:
{recent_actions_text}

Game Memory:
{game_memory}

Strategy Memory:
{strategy_memory}


Respond with only the valid json in the format listed above.
"""
# todo: add back in to tool list
# - send_message: Communicate with other players
        
        logger.debug(f"Generated prompt: {prompt}")
        return prompt