from huggingface_hub import InferenceClient

class RAGGenerator:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        self.client = InferenceClient(model=model_name)

    def generate(self, query, chunks):
        context = "\n\n".join([c.text for c in chunks])

        prompt = f"""
Context:
{context}

Question: {query}
Answer:
"""

        return self.client.text_generation(prompt, max_new_tokens=300)