from dataclasses import dataclass

@dataclass(slots=True)
class GenomeSpec:
    hidden_dim: int = 8
    role_dim: int = 4
    has_goal_state: bool = True
