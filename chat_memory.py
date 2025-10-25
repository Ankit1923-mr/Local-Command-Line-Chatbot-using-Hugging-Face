from collections import deque
from typing import Tuple, List

class ChatMemory:    
    def __init__(self, max_turns: int = 5):
        self.history = deque(maxlen=max_turns)
    
    def add(self, user_input: str, bot_reply: str) -> None:
        self.history.append((user_input.strip(), bot_reply.strip()))
    
    def get_context(self) -> str:
        if not self.history:
            return ""
        
        pieces = []
        for u, b in self.history:
            pieces.append(f"User: {u}\nBot: {b}")
        
        return "\n".join(pieces)
    
    def clear(self) -> None:
        self.history.clear()
    
    def get_turn_count(self) -> int:
        return len(self.history)
    
    def get_history_list(self) -> List[Tuple[str, str]]:
        return list(self.history)
