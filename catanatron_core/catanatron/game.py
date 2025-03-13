"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import List, Union, Optional
import asyncio

from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.state import State, apply_action
from catanatron.state_functions import player_key, player_has_rolled
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, Player
from catanatron.models.message import Message

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def is_valid_action(state, action):
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE in the right time."""
    if action.action_type == ActionType.OFFER_TRADE:
        return (
            state.current_color() == action.color
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, action.color)
            and is_valid_trade(action.value)
        )

    return action in state.playable_actions


def is_valid_trade(action_value):
    """Checks the value of a OFFER_TRADE does not
    give away resources or trade matching resources.
    """
    offering = action_value[:5]
    asking = action_value[5:]
    
    # Must offer exactly one resource and ask for exactly one resource
    if sum(offering) != 1 or sum(asking) != 1:
        return False  # Must be 1:1 trade
        
    # Can't trade same resource type
    offered_index = offering.index(1)
    asked_index = asking.index(1)
    if offered_index == asked_index:
        return False  # Can't trade same resource type
        
    return True


class GameAccumulator:
    """Interface to hook into different game lifecycle events.

    Useful to compute aggregate statistics, log information, etc...
    """

    def __init__(*args, **kwargs):
        pass

    def before(self, game):
        """
        Called when the game is created, no actions have
        been taken by players yet, but the board is decided.
        """
        pass

    def step(self, game_before_action, action):
        """
        Called after each action taken by a player.
        Game should be right before action is taken.
        """
        pass

    def after(self, game):
        """
        Called when the game is finished.

        Check game.winning_color() to see if the game
        actually finished or exceeded turn limit (is None).
        """
        pass


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).
    """

    def __init__(
        self,
        players: List[Player],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        initialize: bool = True,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (List[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            discard_limit (int, optional): Discard limit to use. Defaults to 7.
            vps_to_win (int, optional): Victory Points needed to win. Defaults to 10.
            catan_map (CatanMap, optional): Map to use. Defaults to None.
            initialize (bool, optional): Whether to initialize. Defaults to True.
        """
        if initialize:
            self.seed = seed if seed is not None else random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.state = State(players, catan_map, discard_limit=discard_limit)
            self.message_tasks = []  # Keep track of message handling tasks
            
            # Set game reference in players
            for player in players:
                player.game = self

    async def add_message(self, message: Message):
        """Add a message to the game and notify relevant players"""
        if message.to_color is None:
            # Broadcast message
            tasks = [
                player.receive_message(message) 
                for player in self.state.players
                if player.color != message.from_color
            ]
        else:
            # Direct message
            recipient = next(
                player for player in self.state.players 
                if player.color == message.to_color
            )
            tasks = [recipient.receive_message(message)]

        # Create tasks for message handling
        self.message_tasks.extend([asyncio.create_task(task) for task in tasks])

    async def cleanup_messages(self):
        """Cleanup any pending message tasks"""
        for task in self.message_tasks:
            task.cancel()
        await asyncio.gather(*self.message_tasks, return_exceptions=True)
        self.message_tasks.clear()

    async def play(self, accumulators=[]):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            accumulators (list[Accumulator], optional): list of Accumulator classes to use.
                Their .consume method will be called with every action, and
                their .finalize method will be called when the game ends (if it ends)
                Defaults to [].
        Returns:
            Color: winning color or None if game exceeded TURNS_LIMIT
        """
        for accumulator in accumulators:
            accumulator.before(self)
            
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            current_player = self.state.current_player()
            await current_player.decide(self, self.state.playable_actions)
            
        for accumulator in accumulators:
            accumulator.after(self)
        await self.cleanup_messages()
        return self.winning_color()

    def execute(self, action: Action, validate_action: bool = True) -> Action:
        """Internal call that carries out decided action by player"""
        if validate_action and not is_valid_action(self.state, action):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.state.playable_actions}"
            )

        return apply_action(self.state, action)

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        result = None
        for color in self.state.colors:
            key = player_key(self.state, color)
            if (
                self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
                >= self.vps_to_win
            ):
                result = color

        return result

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game(players=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.state = self.state.copy()
        return game_copy
