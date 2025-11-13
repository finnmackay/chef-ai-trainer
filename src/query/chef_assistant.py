"""Chef AI Assistant - Query interface using RAG."""

from typing import List, Dict, Any
from openai import OpenAI
from langchain_community.vectorstores import Chroma


class ChefAssistant:
    """AI assistant for cooking questions based on chef transcripts."""

    def __init__(
        self,
        vector_store: Chroma,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_k: int = 5
    ):
        """
        Initialize chef assistant.

        Args:
            vector_store: Vector store with chef transcripts
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            top_k: Number of relevant chunks to retrieve
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Initialize OpenAI client
        self.client = OpenAI()

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
        results = self.vector_store.similarity_search_with_score(
            question,
            k=self.top_k
        )

        if verbose:
            print(f"\nRetrieved {len(results)} relevant chunks:")
            for i, (doc, score) in enumerate(results, 1):
                chef = doc.metadata.get('chef', 'Unknown')
                channel = doc.metadata.get('channel', 'Unknown')
                print(f"  {i}. Score: {score:.3f} | Chef: {chef} | Channel: {channel}")

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for doc, score in results:
            chef = doc.metadata.get('chef', 'Unknown')
            channel = doc.metadata.get('channel', 'Unknown')
            video_title = doc.metadata.get('video_title', '')

            context_parts.append(
                f"[Source: {chef} from {channel}]\n{doc.page_content}\n"
            )

            sources.append({
                'chef': chef,
                'channel': channel,
                'video_title': video_title,
                'video_url': doc.metadata.get('video_url', ''),
                'score': float(score)
            })

        context = "\n---\n".join(context_parts)

        # Create user prompt
        user_prompt = f"""Based on the chef transcripts below, answer this cooking question:

Question: {question}

Context from chef transcripts:
{context}

Provide a detailed answer, citing which chef or channel each technique comes from."""

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        answer = response.choices[0].message.content

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'model': self.model
        }

    def interactive(self):
        """Run interactive query session."""
        print("\n" + "="*60)
        print("Chef AI Assistant - Interactive Mode")
        print("="*60)
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
                    print(f"{i}. {source['chef']} ({source['channel']})")
                    if source['video_title']:
                        print(f"   Video: {source['video_title']}")
                    if source['video_url']:
                        print(f"   URL: {source['video_url']}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
