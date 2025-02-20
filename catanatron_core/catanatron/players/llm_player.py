import random
import asyncio
import uuid

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.models.message import Message, MessageType


class LLMPlayer(Player):
    """
    Player that is an LLM powered agent. Uses memory, strategy, and reasoning 
    to decide on the best action to take.
    """
    async def decide(self, game, playable_actions):
        # Process any pending messages
        while not self.message_queue.empty():
            message = self.message_queue.get_nowait()
            await self.handle_message(message)

        # Example: Send a message to all players
        if random.random() < 0.1:  # 10% chance to send message
            message = Message(
                from_color=self.color,
                to_color=None,  # broadcast
                message_type=MessageType.GENERAL,
                content={"text": "Hello everyone!"}
            )
            await self.send_message(game, message)
            
            # Wait for responses with timeout
            await asyncio.sleep(2)  # Give others time to respond

        return random.choice(playable_actions)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.message_type == MessageType.GENERAL:
            # Maybe respond to general messages
            if random.random() < 0.5:  # 50% chance to respond
                response = Message(
                    from_color=self.color,
                    to_color=message.from_color,
                    message_type=MessageType.GENERAL,
                    content={"text": "Hello back!"}
                )
                # We would need game reference to send response
                # await self.send_message(game, response)
                pass
