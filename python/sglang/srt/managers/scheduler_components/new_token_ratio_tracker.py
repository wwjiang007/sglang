from dataclasses import dataclass

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs


@dataclass(slots=True, kw_only=True)
class NewTokenRatioTracker:
    """Scheduler-owned KV-budget headroom factor in ``[min, init]``."""

    init: float
    min: float
    decay: float
    current: float

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "NewTokenRatioTracker":
        init = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        min_ratio = min(
            init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        decay = (init - min_ratio) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        return cls(init=init, min=min_ratio, decay=decay, current=init)

    def decay_step(self) -> None:
        """Decay ``current`` by one step toward ``min``."""
        self.current = max(self.current - self.decay, self.min)

    def reset(self) -> None:
        """Reset ``current`` back to ``init`` (used on scheduler idle)."""
        self.current = self.init
