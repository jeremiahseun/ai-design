"""
Test Advanced Pipeline

Verifies that the integrated pipeline runs correctly.
"""

import os
import sys

# Manual .env loader (in case python-dotenv is not installed)
def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment from {filepath}")
    else:
        print(f"⚠️  {filepath} not found")

# Load environment variables
load_env_file()

from main_pipeline import AdvancedDesignPipeline

def test_pipeline():
    print("="*50)
    print("TESTING ADVANCED DESIGN PIPELINE")
    print("="*50)

    # Check for API Key
    api_key_present = "GOOGLE_API_KEY" in os.environ
    print(f"\n🔑 GOOGLE_API_KEY: {'✅ Found' if api_key_present else '❌ Not found'}")

    if not api_key_present:
        print("   Please check your .env file has GOOGLE_API_KEY set")
        return

    pipeline = AdvancedDesignPipeline()

    # Test Case 1: Simple Brief (No Refs)
    brief = "Make a christmas flyer for a local church. Theme is 'He is Alive'. Date is december 25th by 6pm."
    output_path = "visualizations/test_pipeline_output.png"

    print(f"\n🧪 Running Test Case 1: {brief}")
    try:
        design = pipeline.run(brief=brief, output_path=output_path)

        if design:
            print("✅ Test Case 1 Passed: Image generated.")
            print(f"📁 Saved to: {output_path}")
        else:
            print("❌ Test Case 1 Failed: No image returned.")

    except Exception as e:
        print(f"❌ Test Case 1 Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
