from enum import Enum

class LSLChannel(Enum):
    RELAX = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3
    FEET = 4
    
    @classmethod
    def to_list(cls):
        return [cls.RELAX, cls.LEFT, cls.RIGHT, cls.BOTH, cls.FEET]
    
    @classmethod
    def names(cls):
        return [c.name for c in cls.to_list()]


class StudyClass(Enum):
    REST = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3


class LSLConfig:
    # The prefix for classifier streams. 
    # Complete name will be f"{STREAM_PREFIX}_{ClassifierName}"
    STREAM_PREFIX = "Classifier"
    
    # LSL Content Type
    CONTENT_TYPE = "Probabilities"
    
    # Number of channels (Probabilities for 5 classes)
    CHANNEL_COUNT = 5
    
    # Nominal sampling rate for the output stream
    # It's irregular because it depends on prediction interval, but we set a nominal value.
    NOMINAL_SRATE = 0.0 # 0.0 indicates irregular sampling


class ErrPConfig:
    CONTENT_TYPE = "ErrP_Detection"
    CHANNEL_COUNT = 2  # [p_correct, p_error]
