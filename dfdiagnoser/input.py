import dataclasses as dc


@dc.dataclass
class CheckpointInput:
    checkpoint_dir: str
