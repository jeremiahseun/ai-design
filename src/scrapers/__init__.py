"""
Design Dataset Collection System

A complete toolkit for scraping and labeling professional design images
from Figma Community and Pinterest for training the DTF decoder.

Main components:
- Config: Configuration and API key management
- FigmaScraper: Scrape designs from Figma Community
- PinterestScraper: Scrape designs from Pinterest
- MetadataLabeler: Free heuristic-based labeling
- AILabeler: High-accuracy Claude Vision labeling
- LabelPipeline: Complete end-to-end pipeline

Quick start:
    from src.scrapers import LabelPipeline, load_config

    config = load_config()
    pipeline = LabelPipeline(config)
    pipeline.run(
        figma_file_keys=['...'],
        pinterest_urls=['...'],
        output_dir='scraped_data'
    )
"""

from .config import Config, load_config
from .figma_scraper import FigmaScraper
from .pinterest_scraper import PinterestScraper
from .metadata_labeler import MetadataLabeler, quick_label
from .ai_labeler import AILabeler, quick_label_with_ai
from .label_pipeline import LabelPipeline, run_from_files

__all__ = [
    'Config',
    'load_config',
    'FigmaScraper',
    'PinterestScraper',
    'MetadataLabeler',
    'quick_label',
    'AILabeler',
    'quick_label_with_ai',
    'LabelPipeline',
    'run_from_files',
]

__version__ = '1.0.0'
