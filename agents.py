import os

import requests


class Agent:
    def __init__(self, name : str):
        self.name = name
    
    def run(self, state : dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not imlplemented") 
    

class Generator(Agent):

    def __init__(self, chat_fn):
        super().__init__("Generator")
        self.chat_fn = chat_fn
        
    
    def run(self, state : dict) -> dict:
        context = "\n".join(f" - {text}" for text, _ in state["chunks"])
        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use only the following pieces of context to answer the question. "
            "Don't make up any new information:\n"
            f"{context}\n"
        )
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": state["query"]},
        ]
        answer, model_used = self.chat_fn(messages)
        state["answer"] = answer
        state["model_used"] = model_used
        return state
    
class Evaluator(Agent):
    def __init__(self, min_score : float = 0.5, max_attempts : int = 3) :
        super().__init__("Evaluator")
        self.min_score = min_score
        self.max_attempts = max_attempts
    
    def _score(self, answer : str = "", chunks : list = []) -> float:
        score = 0.5
        match len(answer):
            case 0:
                score = 0.0
            case _:
                score = 1.0
                
        return score
    
    def run(self, state: dict) -> dict:
        state["attempts"] += 1  # increment first
        state["score"] = self._score(state["answer"], state["chunks"])
        state["should_retry"] = (state["score"] < self.min_score) and (state["attempts"] < self.max_attempts)
        return state