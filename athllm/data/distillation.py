"""Offline teacher-distillation data contract.

Teacher outputs are filtered by independent verification before becoming student
training data. This module does not call a proprietary teacher or store weights.
"""
from dataclasses import dataclass

@dataclass
class Candidate:
    prompt: str
    response: str
    task_type: str
    teacher: str
    metadata: dict

@dataclass
class VerifiedExample:
    candidate: Candidate
    reward: float
    verifier: str

def accept(candidate: Candidate, reward: float, verifier: str, minimum: float = 1.0):
    if reward < minimum:
        return None
    if not verifier:
        raise ValueError("Independent verifier is required")
    return VerifiedExample(candidate, reward, verifier)
