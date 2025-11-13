"""Load and process chef transcripts from Obsidian vault."""

import os
from pathlib import Path
from typing import List, Dict, Any
import frontmatter
from tqdm import tqdm


class ObsidianLoader:
    """Load markdown files from Obsidian vault."""

    def __init__(self, vault_path: str, folders: List[str]):
        """
        Initialize Obsidian loader.

        Args:
            vault_path: Path to Obsidian vault
            folders: List of folders to search within vault
        """
        self.vault_path = Path(vault_path)
        self.folders = folders

        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all markdown documents from specified folders.

        Returns:
            List of documents with content and metadata
        """
        documents = []

        for folder in self.folders:
            folder_path = self.vault_path / folder

            if not folder_path.exists():
                print(f"Warning: Folder not found: {folder_path}")
                continue

            # Find all markdown files
            md_files = list(folder_path.rglob("*.md"))

            print(f"Loading {len(md_files)} files from {folder}...")

            for file_path in tqdm(md_files, desc=f"Processing {folder}"):
                try:
                    doc = self._load_document(file_path, folder)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(documents)} documents total")
        return documents

    def _load_document(self, file_path: Path, source_folder: str) -> Dict[str, Any]:
        """
        Load a single markdown document.

        Args:
            file_path: Path to markdown file
            source_folder: Source folder name

        Returns:
            Document dict with content and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            # Parse frontmatter and content
            post = frontmatter.load(f)

        # Extract metadata
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'source_folder': source_folder,
            'chef': post.get('chef', 'Unknown'),
            'channel': post.get('channel', 'Unknown'),
            'video_title': post.get('video_title', ''),
            'video_url': post.get('video_url', ''),
            'published_date': post.get('published_date', ''),
            'tags': post.get('tags', [])
        }

        # Get content
        content = post.content.strip()

        if not content:
            return None

        return {
            'content': content,
            'metadata': metadata
        }

    def get_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.

        Args:
            documents: List of loaded documents

        Returns:
            Statistics dictionary
        """
        total_docs = len(documents)
        total_chars = sum(len(doc['content']) for doc in documents)

        # Count by chef/channel
        chefs = {}
        for doc in documents:
            chef = doc['metadata'].get('chef', 'Unknown')
            chefs[chef] = chefs.get(chef, 0) + 1

        channels = {}
        for doc in documents:
            channel = doc['metadata'].get('channel', 'Unknown')
            channels[channel] = channels.get(channel, 0) + 1

        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'avg_chars_per_doc': total_chars // total_docs if total_docs > 0 else 0,
            'by_chef': chefs,
            'by_channel': channels
        }
