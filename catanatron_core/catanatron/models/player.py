import random
from enum import Enum
import asyncio
from typing import Optional

from catanatron.models.message import Message
from catanatron.models.enums import ActionType, Color


class Player:
    """Interface to represent a player's decision logic.

    Formulated as a class (instead of a function) so that players
    can have an initialization that can later be serialized to
    the database via pickle.
    """

    def __init__(self, color, is_bot=True):
        """Initialize the player

        Args:
            color(Color): the color of the player
            is_bot(bool): whether the player is controlled by the computer
        """
        self.color = color
        self.is_bot = is_bot
        self.message_queue = asyncio.Queue()
        self.pending_responses = {}  # message_id -> Future
        self.game = None

    async def decide(self, game, playable_actions):
        """Should handle the player's turn until they choose to end it.
        
        This method should use execute_action() to perform actions until
        choosing to end the turn.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): initial options
        """
        raise NotImplementedError

    async def execute_action(self, action):
        """Execute an action in the game
        
        Args:
            action (Action): The action to execute
            
        Returns:
            Action: The executed action
            list[Action]: New playable actions
        """
        if not self.game:
            raise ValueError("Player not in a game")
            
        executed_action = self.game.execute(action)
        return executed_action, self.game.state.playable_actions

    async def send_message(self, game, message):
        """Send a message to another player(s)"""
        await game.add_message(message)

    async def wait_for_response(self, message_id, timeout=5.0) -> Optional[Message]:
        """Wait for a response to a specific message"""
        future = asyncio.Future()
        self.pending_responses[message_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending_responses.pop(message_id, None)

    async def receive_message(self, message: Message):
        """Handle incoming messages
        
        Override this method to implement custom message handling.
        Base implementation queues messages for processing.
        """
        await self.message_queue.put(message)

    def reset_state(self):
        """Hook for resetting state between games"""
        self.message_queue = asyncio.Queue()
        self.pending_responses.clear()
        self.game = None

    def __repr__(self):
        return f"{type(self).__name__}:{self.color.value}"


class SimplePlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    async def decide(self, game, playable_actions):
        return playable_actions[0]


class HumanPlayer(Player):
    """Human player that selects which action to take using standard input"""

    async def decide(self, game, playable_actions):
        action = None
        while action is not ActionType.END_TURN:
            for i, action in enumerate(playable_actions):
                print(f"{i}: {action.action_type} {action.value}")
            i = None
            while i is None or (i < 0 or i >= len(playable_actions)):
                print("Please enter a valid index:")
                try:
                    x = input(">>> ")
                    i = int(x)
                except ValueError:
                    pass

        return playable_actions[i]


class RandomPlayer(Player):
    """Random AI player that selects an action randomly from the list of playable_actions"""

    async def decide(self, game, playable_actions):
        while True:
            action = random.choice(playable_actions)
            executed_action, playable_actions = await self.execute_action(action)
            
            # If only option is to end turn, or chose to end turn, break
            if (len(playable_actions) == 1 and 
                playable_actions[0].action_type == ActionType.END_TURN):
                await self.execute_action(playable_actions[0])
                break
            elif action.action_type == ActionType.END_TURN:
                break
