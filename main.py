#!/usr/bin/env python3
"""Chef AI Trainer - Main entry point."""

import os
import sys
from pathlib import Path
import yaml
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest.obsidian_loader import ObsidianLoader
from embeddings.vectorizer import ChefVectorizer
from query.chef_assistant import ChefAssistant

# Load environment variables
load_dotenv()

console = Console()


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, 'r') as f:
        config_str = f.read()

    # Replace environment variables
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", str(value))

    config = yaml.safe_load(config_str)
    return config


@click.group()
def cli():
    """Chef AI Trainer - Train LLMs on professional chef transcripts."""
    pass


@cli.command()
def ingest():
    """Ingest transcripts from Obsidian vault."""
    console.print("\n[bold blue]Ingesting Chef Transcripts[/bold blue]\n")

    config = load_config()

    # Load documents
    loader = ObsidianLoader(
        vault_path=config['obsidian']['vault_path'],
        folders=config['obsidian']['folders']
    )

    documents = loader.load_documents()

    # Show statistics
    stats = loader.get_statistics(documents)

    table = Table(title="Ingestion Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Documents", str(stats['total_documents']))
    table.add_row("Total Characters", f"{stats['total_characters']:,}")
    table.add_row("Avg Chars/Doc", f"{stats['avg_chars_per_doc']:,}")

    console.print(table)

    # Show breakdown by chef
    chef_table = Table(title="Documents by Chef")
    chef_table.add_column("Chef", style="cyan")
    chef_table.add_column("Count", style="magenta")

    for chef, count in sorted(stats['by_chef'].items()):
        chef_table.add_row(chef, str(count))

    console.print("\n")
    console.print(chef_table)

    console.print(f"\n[green]✓ Successfully ingested {stats['total_documents']} documents[/green]\n")


@cli.command()
def embed():
    """Create embeddings from ingested transcripts."""
    console.print("\n[bold blue]Creating Vector Embeddings[/bold blue]\n")

    config = load_config()

    # Load documents
    loader = ObsidianLoader(
        vault_path=config['obsidian']['vault_path'],
        folders=config['obsidian']['folders']
    )

    documents = loader.load_documents()

    # Create vectorizer
    vectorizer = ChefVectorizer(
        embedding_model=config['embedding']['model'],
        chunk_size=config['embedding']['chunk_size'],
        chunk_overlap=config['embedding']['chunk_overlap'],
        vector_db_path=config['vector_db']['path']
    )

    # Create chunks
    chunks = vectorizer.create_chunks(documents)

    # Create vector store
    collection = vectorizer.create_vector_store(
        chunks,
        collection_name=config['vector_db']['collection_name']
    )

    console.print(f"\n[green]✓ Created vector store with {len(chunks)} chunks[/green]\n")


@cli.command()
@click.argument('question')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def query(question, verbose):
    """Query the Chef AI with a cooking question."""
    config = load_config()

    # Load vectorizer
    vectorizer = ChefVectorizer(
        embedding_model=config['embedding']['model'],
        vector_db_path=config['vector_db']['path']
    )

    # Load vector store
    collection = vectorizer.load_vector_store(
        collection_name=config['vector_db']['collection_name']
    )

    # Create assistant
    assistant = ChefAssistant(
        collection=collection,
        vectorizer=vectorizer,
        provider=config['llm']['provider'],
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens'],
        top_k=config['rag']['retrieval']['top_k']
    )

    # Query
    result = assistant.query(question, verbose=verbose)

    # Display result
    console.print("\n[bold]Question:[/bold]", question)
    console.print("\n[bold blue]Answer:[/bold blue]")
    console.print(result['answer'])

    console.print("\n[bold]Sources:[/bold]")
    for i, source in enumerate(result['sources'], 1):
        console.print(f"  {i}. {source['chef']} ({source['channel']})", style="cyan")
        if source['video_title']:
            console.print(f"     Video: {source['video_title']}", style="dim")

    console.print()


@cli.command()
def interactive():
    """Run interactive query session."""
    config = load_config()

    # Load vectorizer
    vectorizer = ChefVectorizer(
        embedding_model=config['embedding']['model'],
        vector_db_path=config['vector_db']['path']
    )

    # Load vector store
    try:
        collection = vectorizer.load_vector_store(
            collection_name=config['vector_db']['collection_name']
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Run 'python main.py embed' first to create the vector database.[/yellow]")
        return

    # Create assistant
    assistant = ChefAssistant(
        collection=collection,
        vectorizer=vectorizer,
        provider=config['llm']['provider'],
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens'],
        top_k=config['rag']['retrieval']['top_k']
    )

    # Run interactive session
    assistant.interactive()


@cli.command()
def status():
    """Show status of Chef AI Trainer."""
    config = load_config()

    table = Table(title="Chef AI Trainer Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="dim")

    # Check Obsidian vault
    vault_path = Path(config['obsidian']['vault_path'])
    if vault_path.exists():
        table.add_row("Obsidian Vault", "✓ Found", str(vault_path))
    else:
        table.add_row("Obsidian Vault", "✗ Not Found", str(vault_path))

    # Check vector database
    vector_db_path = Path(config['vector_db']['path'])
    if vector_db_path.exists():
        table.add_row("Vector Database", "✓ Exists", str(vector_db_path))
    else:
        table.add_row("Vector Database", "✗ Not Created", "Run 'python main.py embed'")

    # Check LLM provider
    provider = config['llm']['provider']
    if provider == "ollama":
        table.add_row("LLM Provider", "Ollama (Local)", f"Model: {config['llm']['model']}")
        table.add_row("API Keys", "Not Required", "Using local models")
    else:
        if os.getenv('OPENAI_API_KEY'):
            table.add_row("LLM Provider", "OpenAI (API)", f"Model: {config['llm']['model']}")
            table.add_row("OpenAI API Key", "✓ Configured", "")
        else:
            table.add_row("LLM Provider", "OpenAI (API)", f"Model: {config['llm']['model']}")
            table.add_row("OpenAI API Key", "✗ Missing", "Set in .env file")

    # Check embedding model
    table.add_row("Embedding Model", "Sentence Transformers", config['embedding']['model'])

    console.print("\n")
    console.print(table)
    console.print("\n")


if __name__ == "__main__":
    cli()
