from dataclasses import dataclass

@dataclass(slots=True)
class ClockSpec:
    fast_every: int = 1
    medium_every: int = 4
    slow_every: int = 16

    def due(self, step: int) -> dict[str, bool]:
        return {
            'fast': step % self.fast_every == 0,
            'medium': step % self.medium_every == 0,
            'slow': step % self.slow_every == 0,
        }
