#!/usr/bin/env python3
"""
Clear cache and temporary files after benchmarking
"""

import os
import shutil
import glob
import sys

def clear_python_cache():
    """Clear Python __pycache__ directories"""
    print("🧹 Clearing Python cache...")
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"   ✅ Removed: {cache_dir}")
                count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {cache_dir}: {e}")
    
    if count == 0:
        print("   ℹ️ No Python cache found")
    else:
        print(f"   ✅ Cleared {count} cache directories")

def clear_benchmark_results():
    """Clear benchmark CSV and plot files"""
    print("\n📊 Clearing benchmark results...")
    
    patterns = [
        'benchmark_summary_*.csv',
        'benchmark_detailed_*.csv',
        'benchmark_plot_*.png'
    ]
    
    count = 0
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"   ✅ Removed: {file}")
                count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {file}: {e}")
    
    if count == 0:
        print("   ℹ️ No benchmark results found")
    else:
        print(f"   ✅ Cleared {count} result files")

def clear_ollama_cache():
    """Show instructions for clearing Ollama cache"""
    print("\n🤖 Ollama Cache:")
    print("   ℹ️ Ollama models are stored separately")
    print("   To clear Ollama cache:")
    print("   1. Stop models: ollama stop <model_name>")
    print("   2. Remove models: ollama rm <model_name>")
    print("   3. Or keep models loaded: ollama ps")

def show_disk_usage():
    """Show current directory disk usage"""
    print("\n💾 Disk Usage:")
    try:
        total_size = 0
        for root, dirs, files in os.walk('.'):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        print(f"   Current directory: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ Could not calculate: {e}")

def main():
    print("="*60)
    print("🧹 CACHE CLEANUP UTILITY")
    print("="*60)
    
    # Ask for confirmation
    print("\nThis will clear:")
    print("  - Python __pycache__ directories")
    print("  - Benchmark CSV files")
    print("  - Benchmark plot images")
    print("\nOllama models will NOT be removed.")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Cancelled")
        sys.exit(0)
    
    print("\n" + "="*60)
    
    # Clear caches
    clear_python_cache()
    clear_benchmark_results()
    clear_ollama_cache()
    show_disk_usage()
    
    print("\n" + "="*60)
    print("✅ CLEANUP COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
