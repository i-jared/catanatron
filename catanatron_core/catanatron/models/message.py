from dataclasses import dataclass
from typing import Optional
from enum import Enum
from datetime import datetime

from catanatron.models.player import Color

class MessageType(Enum):
    TRADE_PROPOSAL = "TRADE_PROPOSAL"
    TRADE_RESPONSE = "TRADE_RESPONSE"
    GENERAL = "GENERAL"

@dataclass
class Message:
    from_color: Color
    to_color: Optional[Color]  # None means broadcast
    message_type: MessageType
    content: dict
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now() 