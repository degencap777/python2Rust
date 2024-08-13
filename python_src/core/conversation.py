from core.conversation_turn import ConversationTurn

class Conversation():
    """Multiturn conversation for multimodal inference. Hash and serialization are used for caching."""

    def __init__(self, conversation_turns: list[ConversationTurn]) -> None:
        self.turns = conversation_turns