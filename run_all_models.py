#!/usr/bin/env python3
"""
Quick script to benchmark all 5 models at once
"""

import subprocess
import sys

# All 4 models
MODELS = ["qwen2.5", "gemma2:2b", "llama3.1", "nxphi47/seallm-7b-v2:q4_0"]

print("🚀 BENCHMARK ALL 4 MODELS")
print("="*80)
print(f"Models: {', '.join(MODELS)}")
print("="*80)

# Ask for number of questions
print("\nHow many questions to test?")
print("  1. Quick test (5 questions, ~2-3 minutes)")
print("  2. Good test (10 questions, ~5-10 minutes)")
print("  3. Better test (20 questions, ~10-20 minutes)")
print("  4. Full HumanEval (164 questions, ~1-2 hours)")

choice = input("\nEnter choice (1-4) or number of questions: ").strip()

if choice == "1":
    num_questions = 5
elif choice == "2":
    num_questions = 10
elif choice == "3":
    num_questions = 20
elif choice == "4":
    num_questions = None  # Full dataset
else:
    try:
        num_questions = int(choice)
    except:
        print("Invalid choice, using 10 questions")
        num_questions = 10

# Build command
cmd = [
    sys.executable,
    "benchmark.py",
    "--models"
] + MODELS + [
    "--dataset", "humaneval"
]

if num_questions:
    cmd.extend(["--num-questions", str(num_questions)])

print(f"\n🎯 Running benchmark with {num_questions if num_questions else 'ALL'} questions...")
print(f"📊 Command: {' '.join(cmd)}")
print("\n" + "="*80)

# Run benchmark
try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\n⚠️ Benchmark interrupted by user")
    sys.exit(1)
