import random
import asyncio
import uuid

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.models.message import Message, MessageType
from .llm_agent import CatanAgent


class LLMPlayer(Player):
    """
    Player that is an LLM powered agent. Uses memory, strategy, and reasoning 
    to decide on the best action to take.
    """
    def __init__(self, color):
        super().__init__(color)
        self.agent = CatanAgent(color)

    async def decide(self, game, playable_actions):
        # Process any pending messages
        while not self.message_queue.empty():
            message = self.message_queue.get_nowait()
            await self.handle_message(message)

        # Convert game state to dict for agent
        game_state = self._get_game_state(game)
        
        # Let agent decide action
        action = await self.agent.decide_action(game_state, playable_actions)
        return action

    async def handle_message(self, message: Message):
        """Handle incoming messages by updating agent memory"""
        self.agent.memory.update_game_memory(
            f"message_{message.from_color}",
            message.content.get("text", "")
        )

    def _get_game_state(self, game) -> dict:
        """Convert game state to dictionary format for agent"""
        # This would need to be implemented based on your game state structure
        return {
            "game_id": id(game),
            "victory_points": game.state.player_state[self.color].victory_points,
            "resources": game.state.player_state[self.color].resource_deck,
            "development_cards": game.state.player_state[self.color].development_deck,
            # Add other relevant game state...
        }
