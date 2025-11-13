"""Chef AI Assistant - Query interface using RAG with Ollama or OpenAI."""

from typing import List, Dict, Any, Optional
import chromadb
import os


class ChefAssistant:
    """AI assistant for cooking questions based on chef transcripts."""

    def __init__(
        self,
        collection: chromadb.Collection,
        vectorizer,
        provider: str = "ollama",
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_k: int = 5
    ):
        """
        Initialize chef assistant.

        Args:
            collection: ChromaDB collection with chef transcripts
            vectorizer: Vectorizer instance for searching
            provider: LLM provider - "ollama" or "openai"
            model: Model name (e.g., "llama3.2" for Ollama, "gpt-4" for OpenAI)
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            top_k: Number of relevant chunks to retrieve
        """
        self.collection = collection
        self.vectorizer = vectorizer
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Initialize LLM client based on provider
        if self.provider == "ollama":
            try:
                import ollama
                self.client = ollama
                print(f"Using Ollama with model: {model}")
            except ImportError:
                raise ImportError("Ollama not installed. Run: pip install ollama")
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI()
                print(f"Using OpenAI with model: {model}")
            except ImportError:
                raise ImportError("OpenAI not installed. Run: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'openai'")

        # System prompt
        self.system_prompt = """You are a professional chef AI assistant trained on techniques from world-class chefs.

Answer cooking questions based ONLY on the provided context from chef transcripts.
If you don't have enough information in the context, say so clearly.
Always cite which chef or channel the information comes from.

Format your responses clearly with:
1. Direct answer to the question
2. Specific techniques or methods mentioned
3. Citations showing which chef/channel provided each tip"""

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Ask a cooking question and get an answer.

        Args:
            question: Cooking question to ask
            verbose: Print detailed information

        Returns:
            Response dict with answer and sources
        """
        # Retrieve relevant chunks
        results = self.vectorizer.search(self.collection, question, self.top_k)

        if verbose:
            print(f"\nRetrieved {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                chef = result['metadata'].get('chef', 'Unknown')
                channel = result['metadata'].get('channel', 'Unknown')
                score = result['score']
                print(f"  {i}. Score: {score:.3f} | Chef: {chef} | Channel: {channel}")

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for result in results:
            chef = result['metadata'].get('chef', 'Unknown')
            channel = result['metadata'].get('channel', 'Unknown')
            video_title = result['metadata'].get('video_title', '')

            context_parts.append(
                f"[Source: {chef} from {channel}]\n{result['content']}\n"
            )

            sources.append({
                'chef': chef,
                'channel': channel,
                'video_title': video_title,
                'video_url': result['metadata'].get('video_url', ''),
                'score': float(result['score'])
            })

        context = "\n---\n".join(context_parts)

        # Create user prompt
        user_prompt = f"""Based on the chef transcripts below, answer this cooking question:

Question: {question}

Context from chef transcripts:
{context}

Provide a detailed answer, citing which chef or channel each technique comes from."""

        # Generate response based on provider
        if self.provider == "ollama":
            answer = self._query_ollama(user_prompt)
        else:
            answer = self._query_openai(user_prompt)

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'model': self.model,
            'provider': self.provider
        }

    def _query_ollama(self, user_prompt: str) -> str:
        """Query using Ollama."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        return response['message']['content']

    def _query_openai(self, user_prompt: str) -> str:
        """Query using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def interactive(self):
        """Run interactive query session."""
        print("\n" + "="*60)
        print("Chef AI Assistant - Interactive Mode")
        print("="*60)
        print(f"Using {self.provider.upper()} with model: {self.model}")
        print("Ask cooking questions based on professional chef transcripts.")
        print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            try:
                question = input("\nYour question: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break

                if not question:
                    continue

                print("\n" + "-"*60)
                result = self.query(question, verbose=True)

                print("\n" + "="*60)
                print("ANSWER:")
                print("="*60)
                print(result['answer'])
                print("\n" + "="*60)
                print("SOURCES:")
                print("="*60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['chef']} ({source['channel']}) - Score: {source['score']:.3f}")
                    if source['video_title']:
                        print(f"   Video: {source['video_title']}")
                    if source['video_url']:
                        print(f"   URL: {source['video_url']}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Make sure Ollama is running (ollama serve) or check your API keys.")
