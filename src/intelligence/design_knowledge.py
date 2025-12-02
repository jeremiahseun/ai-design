"""
Design Knowledge System

Uses Vector Embeddings (CLIP) to understand design concepts semantically.
Allows querying for "luxury", "tech", "warm", etc. and getting relevant
fonts, colors, and layouts.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import sentence_transformers, handle if missing
try:
    from sentence_transformers import SentenceTransformer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("⚠️ sentence-transformers not found. Vector embeddings disabled.")

@dataclass
class DesignConcept:
    category: str  # "font", "color", "layout"
    value: str     # The actual value (e.g., "Didot", "#000000", "asymmetric")
    description: str # Text description for embedding
    embedding: Optional[np.ndarray] = None

class DesignKnowledge:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.concepts: List[DesignConcept] = []
        self.model = None

        if HAS_CLIP:
            try:
                print(f"🧠 Loading CLIP model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                print("✅ CLIP model loaded.")
            except Exception as e:
                print(f"❌ Failed to load CLIP model: {e}")

        self._initialize_knowledge_base()

        # Pre-compute embeddings if model is loaded
        if self.model:
            self._compute_embeddings()

    def _initialize_knowledge_base(self):
        """Define the seed knowledge base."""
        # Fonts
        self.concepts.extend([
            DesignConcept("font", "Didot", "luxury elegant serif high-fashion traditional premium"),
            DesignConcept("font", "Bodoni", "modern serif sharp contrast vogue editorial"),
            DesignConcept("font", "Garamond", "classic literary readable bookish timeless"),
            DesignConcept("font", "Helvetica", "neutral modern clean swiss objective corporate"),
            DesignConcept("font", "Futura", "geometric modern bauhaus forward tech"),
            DesignConcept("font", "Gill Sans", "humanist sans friendly british warm"),
            DesignConcept("font", "Courier", "code typewriter technical brutalist raw"),
            DesignConcept("font", "Brush Script", "handwritten casual personal playful"),
        ])

        # Colors (Hex + Description)
        self.concepts.extend([
            DesignConcept("color", "#000000", "black luxury mystery power sophisticated"),
            DesignConcept("color", "#FFFFFF", "white clean minimal pure space"),
            DesignConcept("color", "#FF0000", "red energetic passion danger bold urgent"),
            DesignConcept("color", "#0000FF", "blue trust corporate calm tech reliable"),
            DesignConcept("color", "#00FF00", "green nature growth fresh eco money"),
            DesignConcept("color", "#FFFF00", "yellow happy caution bright energetic"),
            DesignConcept("color", "#800080", "purple royal creative luxury spiritual"),
            DesignConcept("color", "#FFC0CB", "pink feminine soft romantic sweet"),
            DesignConcept("color", "#FFA500", "orange friendly creative energetic warm"),
            DesignConcept("color", "#A52A2A", "brown earth rustic warm organic"),
        ])

        # Layouts
        self.concepts.extend([
            DesignConcept("layout", "magazine_grid", "editorial fashion complex overlapping text image heavy"),
            DesignConcept("layout", "modern_clean", "minimal whitespace structured corporate balanced"),
            DesignConcept("layout", "split_vertical", "comparison duality side-by-side balanced"),
            DesignConcept("layout", "asymmetric_editorial", "dynamic artistic off-balance creative modern"),
            DesignConcept("layout", "central_hero", "focus spotlight single-subject poster impact"),
            DesignConcept("layout", "diagonal_split", "movement dynamic action sport energy"),
        ])

    def _compute_embeddings(self):
        """Compute embeddings for all concepts."""
        descriptions = [c.description for c in self.concepts]
        embeddings = self.model.encode(descriptions)

        for i, concept in enumerate(self.concepts):
            concept.embedding = embeddings[i]

    def find_nearest(self, query: str, category: Optional[str] = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find concepts nearest to the query string.

        Args:
            query: The search query (e.g., "tech startup")
            category: Optional filter ("font", "color", "layout")
            top_k: Number of results to return

        Returns:
            List of (value, score) tuples
        """
        if not self.model:
            # Fallback: Keyword matching if no model
            results = []
            query_terms = query.lower().split()
            for concept in self.concepts:
                if category and concept.category != category:
                    continue
                score = sum(1 for term in query_terms if term in concept.description.lower())
                if score > 0:
                    results.append((concept.value, float(score)))

            # If no matches found, return a default/random one to prevent crash
            if not results:
                # Find any concept of this category
                candidates = [c for c in self.concepts if not category or c.category == category]
                if candidates:
                    # Return the first one as default
                    return [(candidates[0].value, 0.0)]
                return []

            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        # Encode query
        query_embedding = self.model.encode(query)

        # Calculate similarities
        results = []
        for concept in self.concepts:
            if category and concept.category != category:
                continue

            if concept.embedding is not None:
                # Cosine similarity
                score = np.dot(query_embedding, concept.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(concept.embedding)
                )
                results.append((concept.value, float(score)))

        # Sort by score desc
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def get_recommendations(self, prompt: str) -> Dict[str, str]:
        """
        Get a full set of recommendations (font, color, layout) for a prompt.
        """
        font = self.find_nearest(prompt, "font", 1)[0][0]
        color = self.find_nearest(prompt, "color", 1)[0][0]
        layout = self.find_nearest(prompt, "layout", 1)[0][0]

        return {
            "font": font,
            "color": color,
            "layout": layout
        }
