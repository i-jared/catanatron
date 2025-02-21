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
        self.agent = CatanAgent(color, provider="google")

    async def decide(self, game, playable_actions):
        # Process any pending messages
        while not self.message_queue.empty():
            message = self.message_queue.get_nowait()
            await self.handle_message(message)

        # Convert game state to dict for agent
        game_state = self._get_game_state(game)
        
        if game.state.is_initial_build_phase:
            # Initial building phase - one settlement + road pair per turn
            while True:
                action = await self.agent.decide_action(game_state, playable_actions)
                if action is None or action not in playable_actions:
                    # If agent returns invalid action, choose first valid action
                    action = playable_actions[0]
                
                executed_action, playable_actions = await self.execute_action(action)
                
                # If we just built a settlement, we must build a road
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    # Get road action from agent
                    road_action = await self.agent.decide_action(self._get_game_state(game), playable_actions)
                    if road_action is None or road_action not in playable_actions:
                        road_action = playable_actions[0]  # Should be road action
                    executed_action, playable_actions = await self.execute_action(road_action)
                break
        else:
            # Normal turn - let agent make decisions until it chooses to end turn
            while True:
                action = await self.agent.decide_action(game_state, playable_actions)
                if action is None or action not in playable_actions:
                    # If agent returns invalid action, choose first valid action
                    action = playable_actions[0]
                
                executed_action, playable_actions = await self.execute_action(action)
                
                # Update game state for next decision
                game_state = self._get_game_state(game)
                
                # If only option is to end turn, or agent chose to end turn, break
                if (len(playable_actions) == 1 and 
                    playable_actions[0].action_type == ActionType.END_TURN):
                    await self.execute_action(playable_actions[0])
                    break
                elif action.action_type == ActionType.END_TURN:
                    break

    async def handle_message(self, message: Message):
        """Handle incoming messages by updating agent memory"""
        self.agent.memory.update_game_memory(
            f"message_{message.from_color}",
            message.content.get("text", "")
        )

    def _get_game_state(self, game) -> dict:
        """Convert game state to dictionary format for agent"""
        # Find player number based on position in game's player list
        player_num = game.state.players.index(self)
        prefix = f"P{player_num}_"
        
        # Get all state keys for this player
        player_state = {
            k[len(prefix):]: v 
            for k, v in game.state.player_state.items() 
            if k.startswith(prefix)
        }
        
        # Get recent actions by other players
        recent_actions = []
        if len(game.state.actions) > 0:
            # Start from the most recent action
            actions_iter = reversed(game.state.actions)
            last_turn_index = None
            seen_colors = set()
            current_color = None
            num_players = len(game.state.players)
            
            for action in actions_iter:
                # If we haven't seen a turn index yet, set it
                if last_turn_index is None:
                    last_turn_index = game.state.current_turn_index
                    
                # If we've moved to a previous turn, stop
                current_player_index = game.state.colors.index(action.color)
                if current_player_index > last_turn_index:
                    break
                
                # If this is a new color and we've already seen it before, skip
                if action.color != current_color and action.color in seen_colors:
                    continue
                
                # Add the action and update tracking
                recent_actions.append(action)
                seen_colors.add(action.color)
                current_color = action.color
                
                # If we've seen actions from all other players, stop
                if len(seen_colors) == num_players - 1: 
                    break
            
            # Reverse the list so actions are in chronological order
            recent_actions.reverse()
        
        # Structure the state data in a more logical way
        return {
            "game_id": id(game),
            "player_number": player_num,
            "recent_actions": recent_actions,
            "victory_points": {
                "total": player_state["VICTORY_POINTS"],
                "actual": player_state["ACTUAL_VICTORY_POINTS"],
                "development_cards": player_state["VICTORY_POINT_IN_HAND"]
            },
            "resources": {
                "wood": player_state["WOOD_IN_HAND"],
                "brick": player_state["BRICK_IN_HAND"],
                "sheep": player_state["SHEEP_IN_HAND"],
                "wheat": player_state["WHEAT_IN_HAND"],
                "ore": player_state["ORE_IN_HAND"]
            },
            "development_cards": {
                "knight": {
                    "in_hand": player_state["KNIGHT_IN_HAND"],
                    "played": player_state["PLAYED_KNIGHT"],
                    "owned_at_start": player_state["KNIGHT_OWNED_AT_START"]
                },
                "victory_point": {
                    "in_hand": player_state["VICTORY_POINT_IN_HAND"],
                    "played": player_state["PLAYED_VICTORY_POINT"]
                },
                "year_of_plenty": {
                    "in_hand": player_state["YEAR_OF_PLENTY_IN_HAND"],
                    "played": player_state["PLAYED_YEAR_OF_PLENTY"],
                    "owned_at_start": player_state["YEAR_OF_PLENTY_OWNED_AT_START"]
                },
                "monopoly": {
                    "in_hand": player_state["MONOPOLY_IN_HAND"],
                    "played": player_state["PLAYED_MONOPOLY"],
                    "owned_at_start": player_state["MONOPOLY_OWNED_AT_START"]
                },
                "road_building": {
                    "in_hand": player_state["ROAD_BUILDING_IN_HAND"],
                    "played": player_state["PLAYED_ROAD_BUILDING"],
                    "owned_at_start": player_state["ROAD_BUILDING_OWNED_AT_START"]
                }
            },
            "buildings": {
                "roads_available": player_state["ROADS_AVAILABLE"],
                "settlements_available": player_state["SETTLEMENTS_AVAILABLE"],
                "cities_available": player_state["CITIES_AVAILABLE"],
                "longest_road_length": player_state["LONGEST_ROAD_LENGTH"],
                "has_longest_road": player_state["HAS_ROAD"],
                "has_largest_army": player_state["HAS_ARMY"]
            },
            "turn_state": {
                "has_rolled": player_state["HAS_ROLLED"],
                "has_played_development_card": player_state["HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
            }
        }
