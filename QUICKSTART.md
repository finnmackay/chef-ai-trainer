# Chef AI Trainer - Quick Start Guide

Get up and running with Chef AI Trainer in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Obsidian vault with chef transcripts

## Step 1: Clone and Setup

```bash
# Clone repository
cd /Users/finnmackay/repos
git clone https://github.com/finnmackay/chef-ai-trainer.git
cd chef-ai-trainer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Required settings:
- `OBSIDIAN_VAULT_PATH`: Path to your Obsidian vault
- `OPENAI_API_KEY`: Your OpenAI API key

Example:
```
OBSIDIAN_VAULT_PATH=/Users/finnmackay/Documents/Obsidian Vault
OPENAI_API_KEY=sk-...your-key-here
```

## Step 3: Verify Setup

```bash
python main.py status
```

This will check:
- ✓ Obsidian vault is accessible
- ✓ API keys are configured
- Vector database status

## Step 4: Ingest Transcripts

```bash
python main.py ingest
```

This will:
- Load all markdown files from your chef transcript folders
- Display statistics about loaded documents
- Show breakdown by chef/channel

## Step 5: Create Embeddings

```bash
python main.py embed
```

This will:
- Split transcripts into chunks
- Generate embeddings using OpenAI
- Create and persist vector database

**Note:** This may take a few minutes depending on the number of transcripts.

## Step 6: Query Your Chef AI!

### Single Question
```bash
python main.py query "What's the best way to cook a filet mignon?"
```

### Interactive Mode
```bash
python main.py interactive
```

In interactive mode, you can ask multiple questions:
```
Your question: How do I make hollandaise sauce?
Your question: What temperature for sous vide chicken?
Your question: exit
```

## Example Questions

Try these questions with your Chef AI:

- "What's the best way to cook a filet mignon?"
- "How does Fallow prepare their signature dishes?"
- "What are Chef Steps' tips for sous vide cooking?"
- "How do I make perfect hollandaise sauce?"
- "What's the best temperature for cooking steak?"
- "How do I prevent scrambled eggs from being watery?"

## Folder Structure for Transcripts

Your Obsidian vault should be organized like this:

```
Obsidian Vault/
└── youtube_recipes/
    ├── fallow/
    │   ├── video-1.md
    │   ├── video-2.md
    │   └── ...
    └── chef_steps/
        ├── video-1.md
        ├── video-2.md
        └── ...
```

## Transcript Format

Each markdown file should have frontmatter:

```markdown
---
chef: "Jack Stein"
channel: "Fallow"
video_title: "How to Cook the Perfect Filet Mignon"
video_url: "https://youtube.com/watch?v=..."
published_date: "2024-01-15"
tags: [beef, steak, cooking]
---

[Transcript content here...]
```

## Adding New Chef Channels

1. Add transcripts to a new folder in your Obsidian vault
2. Update `config/config.yaml`:

```yaml
obsidian:
  folders:
    - "youtube_recipes/fallow"
    - "youtube_recipes/chef_steps"
    - "youtube_recipes/new_channel"  # Add here
```

3. Re-run:
```bash
python main.py ingest
python main.py embed
```

## Troubleshooting

**Vector database not found:**
```bash
python main.py embed
```

**API key errors:**
- Check `.env` file has correct `OPENAI_API_KEY`
- Verify key is active on OpenAI platform

**No transcripts found:**
- Verify `OBSIDIAN_VAULT_PATH` in `.env`
- Check folder names in `config/config.yaml` match your vault structure

**Poor quality answers:**
- Try increasing `top_k` in `config/config.yaml` (retrieval section)
- Ensure transcripts have good metadata in frontmatter

## Next Steps

- Add more chef transcripts to your Obsidian vault
- Experiment with different LLM models in config
- Adjust chunk size and retrieval parameters
- Build a web interface (coming soon!)

## Get Help

- Check the [README](README.md) for detailed documentation
- Review `config/config.yaml` for all configuration options
- Open an issue on GitHub for bugs or feature requests
