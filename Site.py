import numpy as np
from params import *

class Site:
    def __init__(self, site_id, quality, pos):
        self.id = site_id
        self.quality = quality
        self.agents_assigned = 0
        self.pos = pos

    def visit(self, agent_id):
        self.agents_assigned += 1
