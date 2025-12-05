"""
Quick API Key Verification Script
Tests if your Gemini API key is working correctly
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

print("="*60)
print("GEMINI API KEY VERIFICATION")
print("="*60)

# Check if key exists
api_key = os.environ.get('GOOGLE_API_KEY', '').strip()

print(f"\n1️⃣  Checking .env file...")
if api_key:
    print(f"   ✅ GOOGLE_API_KEY found")
    print(f"   📋 Key preview: {api_key[:10]}...{api_key[-4:]}")
    print(f"   📏 Length: {len(api_key)} characters")
else:
    print(f"   ❌ GOOGLE_API_KEY not found in environment")
    sys.exit(1)

# Try to import and configure
print(f"\n2️⃣  Testing Google Generative AI library...")
try:
    import google.generativeai as genai
    print(f"   ✅ Library imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import: {e}")
    sys.exit(1)

# Configure with API key
print(f"\n3️⃣  Configuring API...")
try:
    genai.configure(api_key=api_key)
    print(f"   ✅ API configured")
except Exception as e:
    print(f"   ❌ Configuration failed: {e}")
    sys.exit(1)

# Make a simple test call
print(f"\n4️⃣  Making test API call...")
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content("Say 'Hello' if you can hear me")

    print(f"   ✅ API call successful!")
    print(f"   📝 Response: {response.text[:100]}")

except Exception as e:
    print(f"   ❌ API call failed: {e}")
    print(f"\n💡 Common issues:")
    print(f"   - Key has extra spaces/quotes")
    print(f"   - Key is expired or invalid")
    print(f"   - API quota exceeded")
    print(f"   - Wrong key type (need Gemini AI Studio key)")
    sys.exit(1)

print(f"\n" + "="*60)
print("✅ SUCCESS - Your API key is working correctly!")
print("="*60)
