from dataclasses import dataclass, field

@dataclass
class Shot:
    sequence: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    features: dict = field(default_factory=lambda : {})
    test_is_ad: bool = False
    is_ad: bool = None
    
    @property
    def duration(self):
        return self.end_timestamp - self.start_timestamp