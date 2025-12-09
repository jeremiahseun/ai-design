"""
Configuration management for scraping and labeling system.

This module handles API tokens and configuration settings.
Create a config.json file in this directory with your credentials.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Manages configuration and API credentials."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.json file. If None, looks in current directory.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from JSON file or environment variables."""
        config = {
            'figma': {
                'access_token': None,
                'rate_limit_delay': 1.0,  # seconds between requests
            },
            'gemini': {
                'api_key': None,
                'model': 'gemini-2.5-flash-lite',
            },
            'pinterest': {
                'headless': True,  # Run browser in headless mode
                'scroll_delay': 2.5,  # Delay between scrolls (seconds)
                'max_scrolls': 10,  # Maximum scroll iterations
                'timeout': 60000,  # Page load timeout (milliseconds) - Pinterest is slow
                'rate_limit_delay': 2.0,  # Delay between requests (seconds)
            },
            'scraping': {
                'max_images': 5000,
                'target_size': (256, 256),
                'formats': {
                    'poster': 0.30,  # 30% of dataset
                    'social': 0.30,
                    'flyer': 0.20,
                    'banner': 0.20,
                },
                'min_confidence_for_auto_label': 0.7,
            },
            'paths': {
                'raw_data': 'scraped_data/raw',
                'labeled_data': 'scraped_data/labeled',
                'final_dataset': 'scraped_data/final_dataset',
            }
        }

        # Load from file if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
                # Merge file config into defaults (deep merge)
                self._deep_merge(config, file_config)

        # Override with environment variables if present
        if os.getenv('FIGMA_ACCESS_TOKEN'):
            config['figma']['access_token'] = os.getenv('FIGMA_ACCESS_TOKEN')
        if os.getenv('GOOGLE_API_KEY'):
            config['gemini']['api_key'] = os.getenv('GOOGLE_API_KEY')

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, *keys):
        """Get nested configuration value using dot notation."""
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return None
        return value

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that required configuration is present.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check Figma token
        if not self.get('figma', 'access_token'):
            errors.append("Figma access token not configured")

        # Check Gemini API key
        if not self.get('gemini', 'api_key'):
            errors.append("Gemini API key not configured")

        return len(errors) == 0, errors

    def create_example_config(self, output_path: Optional[str] = None) -> None:
        """
        Create an example config.json file.

        Args:
            output_path: Where to save the example. Defaults to config.example.json
        """
        if output_path is None:
            output_path = Path(__file__).parent / "config.example.json"

        example = {
            "figma": {
                "access_token": "YOUR_FIGMA_ACCESS_TOKEN_HERE",
                "rate_limit_delay": 1.0
            },
            "gemini": {
                "api_key": "GEMINI_API_KEY",
                "model": "gemini-2.5-flash-lite"
            },
            "pinterest": {
                "headless": True,
                "scroll_delay": 2.5,
                "max_scrolls": 10,
                "timeout": 30000,
                "rate_limit_delay": 2.0
            },
            "scraping": {
                "max_images": 5000,
                "target_size": [256, 256],
                "formats": {
                    "poster": 0.30,
                    "social": 0.30,
                    "flyer": 0.20,
                    "banner": 0.20
                },
                "min_confidence_for_auto_label": 0.7
            },
            "paths": {
                "raw_data": "scraped_data/raw",
                "labeled_data": "scraped_data/labeled",
                "final_dataset": "scraped_data/final_dataset"
            }
        }

        with open(output_path, 'w') as f:
            json.dump(example, f, indent=2)

        print(f"Example config created at: {output_path}")
        print("\nTo use:")
        print("1. Copy config.example.json to config.json")
        print("2. Add your Figma access token and Claude API key")
        print("3. Adjust settings as needed")


# Convenience function
def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    return Config(config_path)


if __name__ == "__main__":
    # Create example config when run directly
    config = Config()
    config.create_example_config()
