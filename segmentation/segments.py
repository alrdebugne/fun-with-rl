from dataclasses import dataclass

@dataclass
class Segments:
    """ Segments used for SMB segmentation """
    default: int = 0
    floor: int = 1
    brick: int = 2
    box: int = 3
    enemy: int = 4
    mario: int = 5
