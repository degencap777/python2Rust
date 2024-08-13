from core.data_bundle import DataBundle
from core.role import Role

class ConversationTurn():
    """Serializable conversation turn between User and Model."""

    def __init__(self, data_bundle: DataBundle, role: Role = Role.USER) -> None:
        self.role = role
        self.data_bundle = data_bundle