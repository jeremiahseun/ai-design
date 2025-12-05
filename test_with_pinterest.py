"""
Test Advanced Pipeline with Pinterest References

Verifies that the integrated pipeline runs correctly with Pinterest style analysis.
"""

import os
import sys

# Load .env manually
def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print(f"⚠️  {filepath} not found")

load_env_file()

from main_pipeline import AdvancedDesignPipeline
from pinterest_helper import get_pinterest_images

def test_pipeline_with_pinterest():
    print("="*60)
    print("TESTING ADVANCED DESIGN PIPELINE")
    print("WITH PINTEREST STYLE ANALYSIS")
    print("="*60)

    # Check for API Key
    api_key_present = "GOOGLE_API_KEY" in os.environ
    print(f"\n🔑 GOOGLE_API_KEY: {'✅ Found' if api_key_present else '❌ Not found'}")

    if not api_key_present:
        print("   Please check your .env file has GOOGLE_API_KEY set")
        return

    # Get Pinterest reference images
    print(f"\n📸 Loading Pinterest reference images...")
    pinterest_refs = get_pinterest_images(max_images=3)

    if not pinterest_refs:
        print("   ⚠️  No Pinterest images found")
        print("   Falling back to no references")
        pinterest_refs = None
    else:
        print(f"   ✅ Loaded {len(pinterest_refs)} references:")
        for ref in pinterest_refs:
            print(f"      - {os.path.basename(ref)}")

    pipeline = AdvancedDesignPipeline()

    # Test Case: AI Conference with Pinterest style learning
    brief = "Make a christmas flyer for a local church. Theme is 'He is Alive'. Date is december 25th by 6pm."
    output_path = "visualizations/test_pipeline_with_pinterest.png"

    print(f"\n🧪 Running Test: {brief}")
    print(f"📁 Output: {output_path}\n")

    try:
        design = pipeline.run(
            brief=brief,
            pinterest_refs=pinterest_refs,
            output_path=output_path
        )

        if design:
            print("\n" + "="*60)
            print("✅ SUCCES Test Passed!")
            print("="*60)
            print(f"📁 Design saved to: {output_path}")
            print(f"\n💡 View it with: open {output_path}")
        else:
            print("\n❌ Test Failed: No image returned.")

    except Exception as e:
        print(f"\n❌ Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_with_pinterest()
