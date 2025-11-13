# Chef AI Trainer

Train a custom LLM on professional chef YouTube transcripts to create an intelligent cooking assistant.

## Overview

This project ingests cooking transcripts from YouTube channels (Fallow, Chef Steps, etc.) stored in Obsidian and creates an AI assistant that can answer cooking questions based on professional chef techniques and methods.

## Features

- **Transcript Ingestion**: Load and process cooking transcripts from Obsidian vault
- **Vector Embeddings**: Create semantic embeddings of chef techniques and recipes
- **RAG System**: Retrieval-Augmented Generation for accurate chef-based answers
- **Multi-Source**: Support for multiple chef channels (Fallow, Chef Steps, etc.)
- **Query Interface**: Ask cooking questions and get answers based on professional chef knowledge

## Project Structure

```
chef-ai-trainer/
├── src/
│   ├── ingest/          # Data ingestion from Obsidian
│   ├── embeddings/      # Vector embedding generation
│   ├── query/           # Query interface and RAG
│   └── models/          # Model configurations
├── config/              # Configuration files
├── data/                # Processed data and vector store
├── tests/               # Unit tests
└── main.py              # Main entry point
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/finnmackay/chef-ai-trainer.git
cd chef-ai-trainer
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Obsidian vault path and API keys
```

5. Configure data sources:
```bash
# Edit config/config.yaml to specify your chef channels
```

## Usage

### Ingest Transcripts
```bash
python main.py ingest
```

### Create Embeddings
```bash
python main.py embed
```

### Query the Chef AI
```bash
python main.py query "What's the best way to cook a filet mignon?"
```

### Interactive Mode
```bash
python main.py interactive
```

## Example Queries

- "What's the best way to cook a filet mignon?"
- "How does Fallow prepare their signature dish?"
- "What temperature should I use for sous vide chicken?"
- "Chef Steps tips for making perfect hollandaise?"

## Configuration

Edit `config/config.yaml`:

```yaml
obsidian:
  vault_path: "/path/to/your/obsidian/vault"
  folders:
    - "youtube_recipes/fallow"
    - "youtube_recipes/chef_steps"

embedding:
  model: "text-embedding-3-large"
  chunk_size: 1000
  chunk_overlap: 200

llm:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
```

## Data Sources

Currently supported YouTube chef channels:
- **Fallow**: Fine dining techniques and signature dishes
- **Chef Steps**: Precision cooking and modernist cuisine
- *(Extensible to any YouTube chef channel)*

## Roadmap

- [ ] Basic transcript ingestion
- [ ] Vector embedding generation
- [ ] RAG query system
- [ ] Fine-tuning pipeline
- [ ] Web interface
- [ ] Recipe extraction and structuring
- [ ] Multi-modal support (images from videos)
- [ ] Chef technique categorization

## Contributing

Pull requests welcome! Please open an issue first to discuss proposed changes.

## License

MIT
