from dataclasses import dataclass, field


@dataclass
class Config:
    batch_size: int = field(default=32)
    setup_template: bool = field(default=False)
