        
class ProtonException(Exception):
    def __init__(
        self,
        message: str | None = None,
    ):
        super(ProtonException, self).__init__(message)

class InvalidArgsError(ProtonException):
    """A proton method was called with invalid arguments."""

