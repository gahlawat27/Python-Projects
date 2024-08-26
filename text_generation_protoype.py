import openai

class TextGenerator:
    def __init__(self, api_key, model_name="text-davinci-003"):
        self.api_key = api_key
        self.model_name = model_name
        self.openai = openai.OpenAI(api_key=self.api_key)

    def generate_text(self, prompt, max_tokens=500, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        response = self.openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        generated_text = response.choices[0].text
        return generated_text

# Example usage:
if __name__ == "__main__":
    api_key = "your_openai_api_key"
    text_generator = TextGenerator(api_key)

    # Prompt for text generation
    prompt = "Write a 500-word blog post on the benefits of meditation for stress relief."

    # Generate text
    generated_text = text_generator.generate_text(prompt, max_tokens=500, temperature=0.7)

    # Print the generated text
    print(generated_text)