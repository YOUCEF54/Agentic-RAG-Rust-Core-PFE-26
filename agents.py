import re

class Agent:
    def __init__(self, name: str):
        self.name = name
    
    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not implemented") 
    

class Generator(Agent):
    def __init__(self, chat_fn):
        super().__init__("Generator")
        self.chat_fn = chat_fn
        
    def run(self, state: dict) -> dict:
        context = "\n".join(f" - {text}" for text, _ in state["chunks"])
        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use only the following pieces of context to answer the question. "
            "Don't make up any new information:\n"
            f"{context}\n"
        )
        query = state.get("refined_query") or state["query"]
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        answer, model_used = self.chat_fn(messages)
        state["answer"] = answer
        state["model_used"] = model_used
        return state
    

class Evaluator(Agent):
    def __init__(self, chat_fn, min_score: float = 0.7, max_attempts: int = 3):
        super().__init__("Evaluator")
        self.chat_fn = chat_fn
        # On augmente le seuil d'exigence à 0.7 (70% de qualité minimum)
        self.min_score = min_score
        self.max_attempts = max_attempts
    
    def _score(self, query: str, answer: str, chunks: list) -> float:
        if not answer.strip():
            return 0.0
            
        context = "\n".join(f" - {text}" for text, _ in chunks)
        
        # Prompt "LLM-as-a-Judge"
        eval_prompt = (
            "You are a strict grading agent for a RAG system.\n"
            "Evaluate the quality of the 'Generated Answer' based on the 'User Query' and the 'Retrieved Context'.\n\n"
            "Criteria:\n"
            "- Faithfulness: Is the answer derived strictly from the context without hallucinations?\n"
            "- Relevance: Does the answer directly address the user's query?\n\n"
            f"User Query: {query}\n"
            f"Retrieved Context:\n{context}\n"
            f"Generated Answer:\n{answer}\n\n"
            "INSTRUCTIONS:\n"
            "1. First, write a brief 1-sentence reasoning explaining your evaluation.\n"
            "2. Then, on a new line, write 'SCORE: ' followed by a float between 0.0 and 1.0.\n"
        )
        
        messages = [{"role": "user", "content": eval_prompt}]
        
        try:
            llm_response, _ = self.chat_fn(messages)
            lines = llm_response.strip().split("\n")
            reasoning = lines[0] if lines else "No reasoning provided"
            print(f"[{self.name}] Reasoning: {reasoning}")
            # Utilisation d'une regex pour extraire le score flottant de manière robuste
            # au cas où le LLM serait trop bavard (ex: "The score is 0.85")
            # Cherche "SCORE: 0.85" par exemple
            match = re.search(r"SCORE:\s*(0\.\d+|1\.0)", llm_response)
            if match:
                score = float(match.group(1))
            else:
                # Fallback de parsing direct
                score = 0.5 # neutral
                
            # Clamper la valeur entre 0.0 et 1.0 par sécurité
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"[{self.name}] Failed to parse score from LLM response: {e}")
            return 0.0
    
    def run(self, state: dict) -> dict:
        state["attempts"] += 1
        print(f"\n[{self.name}] Scoring attempt {state['attempts']}...")
        
        state["score"] = self._score(state["query"], state["answer"], state["chunks"])
        
        # La condition de retry : score insuffisant ET on a encore des essais disponibles
        state["should_retry"] = (state["score"] < self.min_score) and (state["attempts"] < self.max_attempts)
        return state
    
class QueryRefiner(Agent):
    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        prompt = (
            "Rewrite the user query to be clearer and more specific for retrieval. "
            "Keep the original intent, do not answer the query, and return only the rewritten query.\n\n"
            f"User query: {state['query']}"
        )
        messages = [{"role": "user", "content": prompt}]
        refined_query, _ = self.chat_fn(messages)
        state["refined_query"] = refined_query.strip()
        print(f"[{self.name}] Original query: {state['query']}")
        print(f"[{self.name}] Refined query: {state['refined_query']}")
        return state
