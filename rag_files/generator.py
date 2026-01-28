from ollama import Client

class RAGGenerator:
    def __init__(self, model_name="phi3:mini"):
        from ollama import Client
        self.client = Client()
        self.model = model_name

    def generate(self, query, chunks):
        context = "\n\n".join([c.text for c in chunks])

        prompt = f"""
You are an expert analyst.

Using ONLY the information in the context:
1. Compare all relevant parts.
2. Infer relationships.
3. Synthesize a single precise answer.
4. Do NOT copy sentences verbatim.
5. Cite sources at the end.

Context:
{context}

Question:
{query}

Step-by-step reasoning, then final concise answer:
"""

        response = self.client.generate(
            model=self.model,
            prompt=prompt
        )

        return response["response"]
