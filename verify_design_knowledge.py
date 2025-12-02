
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies to avoid loading heavy libs just for this test if not needed
# But we DO need sentence_transformers if installed.
# We'll assume the user has installed it or we'll fallback gracefully.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.intelligence.design_knowledge import DesignKnowledge

def test_design_knowledge():
    print("="*60)
    print("Testing Design Knowledge System (Vector Embeddings)")
    print("="*60)

    try:
        dk = DesignKnowledge()
    except Exception as e:
        print(f"❌ Failed to initialize DesignKnowledge: {e}")
        return

    if not dk.model:
        print("⚠️  CLIP model not loaded. Running in fallback mode.")
    else:
        print("✅ CLIP model loaded successfully.")

    queries = [
        "luxury fashion brand",
        "tech startup software",
        "organic farm nature",
        "playful kids toy"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)

        recs = dk.get_recommendations(query)
        print(f"  Font:   {recs['font']}")
        print(f"  Color:  {recs['color']}")
        print(f"  Layout: {recs['layout']}")

        # Check specific expectations
        if "luxury" in query:
            if recs['font'] in ["Didot", "Bodoni"]: print("  ✅ Font matches luxury")
            else: print(f"  ⚠️ Font '{recs['font']}' might not be optimal for luxury")

        if "tech" in query:
            if recs['font'] in ["Helvetica", "Futura", "Roboto"]: print("  ✅ Font matches tech")
            if recs['color'] in ["#0000FF", "#000000", "#FFFFFF"]: print("  ✅ Color matches tech")

if __name__ == "__main__":
    test_design_knowledge()
