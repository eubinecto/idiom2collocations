"""
dataclasses. The objects to work with.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Example:
    response: str
    contexts: List[str]

