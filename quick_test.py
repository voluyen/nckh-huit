#!/usr/bin/env python3
"""
Quick test script to verify SeaLLM v2 installation
"""

import subprocess
import sys

print("🧪 QUICK TEST - SeaLLM v2")
print("="*80)

# Check if Ollama is running
print("\n1️⃣ Checking Ollama...")
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "nxphi47/seallm-7b-v2:q4_0" in result.stdout:
        print("   ✅ SeaLLM v2 is installed!")
    else:
        print("   ❌ SeaLLM v2 not found!")
        print("   📥 Install with: ollama pull nxphi47/seallm-7b-v2:q4_0")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Make sure Ollama is running!")
    sys.exit(1)

# Test the model
print("\n2️⃣ Testing SeaLLM v2...")
print("   Asking: 'Xin chào! Bạn là ai?'")
try:
    result = subprocess.run(
        ["ollama", "run", "nxphi47/seallm-7b-v2:q4_0", "Xin chào! Bạn là ai?"],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"   ✅ Response: {result.stdout[:100]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Run quick benchmark
print("\n3️⃣ Running quick benchmark (5 questions)...")
print("   This will take ~2-3 minutes...")
try:
    subprocess.run([
        sys.executable,
        "benchmark.py",
        "--models", "nxphi47/seallm-7b-v2:q4_0",
        "--dataset", "humaneval",
        "--num-questions", "5"
    ])
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\n🎉 SeaLLM v2 is ready to use!")
print("\n📚 Next steps:")
print("   1. Run chatbot: streamlit run chatbot.py")
print("   2. Test all models: python run_all_models.py")
print("   3. Full benchmark: python benchmark.py --models qwen2.5 gemma2:2b llama3.1 nxphi47/seallm-7b-v2:q4_0 --dataset humaneval --num-questions 10")
