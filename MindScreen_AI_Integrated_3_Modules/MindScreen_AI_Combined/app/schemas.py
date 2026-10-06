from dataclasses import dataclass

@dataclass
class PHQ9Result:
    score: int
    severity: str
    q9_positive: bool
