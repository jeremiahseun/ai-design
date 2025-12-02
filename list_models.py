
import os
import google.generativeai as genai

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"\'')

if "GEMINI_API_KEY" not in os.environ:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("🔍 Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")
