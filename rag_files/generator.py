from huggingface_hub import InferenceClient

class RAGGenerator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.client = InferenceClient(model=model_name)
        self.model_name = model_name

    def generate(self, query, chunks):
        context = "\n\n".join([c.text for c in chunks])

        messages = [
            {
                "role": "system",
                "content": "You are an expert analyst. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Give a concise, accurate answer based only on the context.
"""
            }
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=512,
            temperature=0.2
        )

        return response.choices[0].message.content