import dataclasses as dc


@dc.dataclass
class CheckpointInput:
    checkpoint_dir: str


@dc.dataclass
class MofkaInput:
    group_file: str
    topic_name: str
