from dataclasses import dataclass
from morphobase.clocks import ClockSpec

@dataclass(slots=True)
class StepSchedule:
    fast: bool
    medium: bool
    slow: bool

class Scheduler:
    def __init__(self, clock_spec: ClockSpec | None = None):
        self.clock_spec = clock_spec or ClockSpec()
    def due(self, step: int) -> StepSchedule:
        return StepSchedule(**self.clock_spec.due(step))
