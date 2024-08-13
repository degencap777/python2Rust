import os
import logging
from enum import IntEnum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger('nltk_data').setLevel(logging.CRITICAL)  # Reduce noise
logging.getLogger('urllib3').setLevel(logging.CRITICAL)  # Reduce noise

_proton_logger = logging.getLogger('P')
_proton_logger.setLevel(logging.INFO)

class Trace(IntEnum):
    """Trace levels used in Proton."""
    OFF = 0
    ON = 1
    VERBOSE = 2

    def __repr__(self) -> str:
        return self.name

def log(msg, trace: int = 1):
    if trace > 0:
        _proton_logger.log(logging.INFO, msg)

def list_all_loggers( name_prefix: str | None = None ) -> list[str]:
    """Return the list of all registered loggers to help optimize the logging levels"""
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    results = []
    for logger in loggers:
        if name_prefix:
            if logger.name.startswith(name_prefix):
                results.append(logger.name)
        else:
            results.append(logger.name)
    return results