import random

from catanatron.models.player import Player
from catanatron.models.actions import ActionType


WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}


class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    async def decide(self, game, playable_actions):
        while True:
            # Create weighted list of actions
            bloated_actions = []
            for action in playable_actions:
                weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
                bloated_actions.extend([action] * weight)

            # Choose and execute action
            action = random.choice(bloated_actions)
            executed_action, playable_actions = await self.execute_action(action)
            
            # If only option is to end turn, or chose to end turn, break
            if (len(playable_actions) == 1 and 
                playable_actions[0].action_type == ActionType.END_TURN):
                await self.execute_action(playable_actions[0])
                break
            elif action.action_type == ActionType.END_TURN:
                break
