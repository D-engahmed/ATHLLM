"""Three RL tracks with outcome-based reward contracts."""
from dataclasses import dataclass
from typing import Callable
@dataclass
class Rollout: prompt:str; response:str; metadata:dict
Reward=Callable[[Rollout],float]
def reasoning_verifiable(r,verifier): return verifier(r)
def coding_execution(r,executor): return executor(r)
def self_improvement(r,verifier): return verifier(r)
RL_TRACKS={"reasoning_verifiable":reasoning_verifiable,"coding_execution":coding_execution,"self_improvement":self_improvement}
