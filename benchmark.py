#!/usr/bin/env python3
"""
Comprehensive Local Benchmark Runner
Runs benchmarks locally with detailed statistics and visualizations
"""

import argparse
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from tabulate import tabulate
from benchmark_datasets import BenchmarkDatasets, ModelBenchmark

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ matplotlib/seaborn not installed. Plotting disabled.")
    print("   Install with: pip install matplotlib seaborn")

class BenchmarkRunner:
    """Enhanced benchmark runner with statistics and visualization"""
    
    def __init__(self, models: List[str], dataset: str, num_questions: Optional[int] = None):
        self.models = models
        self.dataset = dataset
        self.num_questions = num_questions
        self.benchmark_data = BenchmarkDatasets()
        self.start_time = None
        self.end_time = None
        
    def load_dataset(self) -> List[str]:
        """Load dataset and return questions"""
        print(f"\n📥 Loading dataset: {self.dataset}")
        
        if self.dataset == "vimmrc":
            samples = self.benchmark_data.load_vietnamese_qa_samples(self.num_questions)
        elif self.dataset == "humaneval":
            samples = self.benchmark_data.load_coding_samples(self.num_questions)
        elif self.dataset == "boolq":
            samples = self.benchmark_data.load_boolq_samples(self.num_questions)
        elif self.dataset == "squad":
            samples = self.benchmark_data.load_squad_samples(self.num_questions)
        else:
            samples = self.benchmark_data.load_commonsense_samples(self.num_questions)
        
        questions = [s.get("question", str(s)) for s in samples]
        print(f"✅ Loaded {len(questions)} questions")
        
        return questions
    
    def estimate_time(self, num_questions: int, num_models: int) -> str:
        """Estimate total benchmark time"""
        avg_time_per_question = 5  # seconds
        total_seconds = num_questions * num_models * avg_time_per_question
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds/60:.1f} minutes"
        else:
            return f"{total_seconds/3600:.1f} hours"
    
    def print_header(self, questions: List[str]):
        """Print benchmark header"""
        print("\n" + "="*80)
        print("🚀 LOCAL BENCHMARK RUNNER")
        print("="*80)
        print(f"📊 Dataset: {self.dataset}")
        print(f"🤖 Models: {', '.join(self.models)}")
        print(f"❓ Questions: {len(questions)}")
        print(f"⏱️  Estimated time: {self.estimate_time(len(questions), len(self.models))}")
        print("="*80)
    
    def run_benchmark(self) -> tuple:
        """Run the benchmark and return results"""
        questions = self.load_dataset()
        
        # Confirm if large dataset
        if len(questions) > 100:
            print(f"\n⚠️ WARNING: Testing {len(questions)} questions will take a long time!")
            print(f"   Estimated time: {self.estimate_time(len(questions), len(self.models))}")
            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                print("❌ Cancelled.")
                sys.exit(0)
        
        self.print_header(questions)
        
        # Run benchmark
        self.start_time = time.time()
        
        model_benchmark = ModelBenchmark(self.models)
        summary_df, detailed_df = model_benchmark.run_detailed_benchmark(questions)
        
        self.end_time = time.time()
        
        return summary_df, detailed_df, questions
    
    def print_summary_stats(self, summary_df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("📊 SUMMARY STATISTICS")
        print("="*80)
        
        # Print table
        print("\n" + tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Overall stats
        total_questions = summary_df["Tổng câu hỏi"].iloc[0]
        total_tests = len(self.models) * total_questions
        total_success = summary_df["Thành công"].sum()
        total_failures = summary_df["Thất bại"].sum()
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total tests: {total_tests}")
        print(f"   Total success: {total_success} ({total_success/total_tests*100:.1f}%)")
        print(f"   Total failures: {total_failures} ({total_failures/total_tests*100:.1f}%)")
        
        # Best performers
        print(f"\n🏆 Best Performers:")
        
        # Fastest model
        fastest_idx = summary_df["Thời gian TB (s)"].idxmin()
        fastest_model = summary_df.loc[fastest_idx, "Model"]
        fastest_time = summary_df.loc[fastest_idx, "Thời gian TB (s)"]
        print(f"   ⚡ Fastest: {fastest_model} ({fastest_time}s avg)")
        
        # Most reliable
        reliable_idx = summary_df["Thành công"].idxmax()
        reliable_model = summary_df.loc[reliable_idx, "Model"]
        reliable_count = summary_df.loc[reliable_idx, "Thành công"]
        print(f"   ✅ Most reliable: {reliable_model} ({reliable_count}/{total_questions} success)")
        
        # Most efficient (success rate / time)
        summary_df['efficiency'] = summary_df["Thành công"] / summary_df["Thời gian TB (s)"]
        efficient_idx = summary_df['efficiency'].idxmax()
        efficient_model = summary_df.loc[efficient_idx, "Model"]
        print(f"   🎯 Most efficient: {efficient_model}")
    
    def print_detailed_stats(self, summary_df: pd.DataFrame, detailed_df: pd.DataFrame):
        """Print detailed statistics"""
        print("\n" + "="*80)
        print("📈 DETAILED STATISTICS")
        print("="*80)
        
        for model in self.models:
            model_data = summary_df[summary_df["Model"] == model].iloc[0]
            model_details = detailed_df[detailed_df["Model"] == model]
            
            print(f"\n🤖 {model}:")
            print(f"   Success rate: {model_data['Thành công']}/{model_data['Tổng câu hỏi']} ({model_data['Thành công']/model_data['Tổng câu hỏi']*100:.1f}%)")
            print(f"   Avg time: {model_data['Thời gian TB (s)']}s")
            print(f"   Min time: {model_data['Thời gian Min (s)']}s")
            print(f"   Max time: {model_data['Thời gian Max (s)']}s")
            print(f"   Total time: {model_data['Tổng thời gian (s)']}s ({model_data['Tổng thời gian (s)']/60:.1f} min)")
            
            # Failure analysis
            failures = model_details[model_details["Trạng thái"].str.contains("❌")]
            if len(failures) > 0:
                print(f"   ⚠️ Failures: {len(failures)}")
                # Show first few failures
                for idx, row in failures.head(3).iterrows():
                    print(f"      - {row['Câu hỏi']}: {row['Trạng thái']}")
    
    def print_comparison(self, summary_df: pd.DataFrame):
        """Print model comparison"""
        print("\n" + "="*80)
        print("⚖️ MODEL COMPARISON")
        print("="*80)
        
        # Speed comparison
        print("\n⚡ Speed Ranking (fastest to slowest):")
        speed_sorted = summary_df.sort_values("Thời gian TB (s)")
        for idx, row in speed_sorted.iterrows():
            print(f"   {idx+1}. {row['Model']}: {row['Thời gian TB (s)']}s")
        
        # Reliability comparison
        print("\n✅ Reliability Ranking (most to least successful):")
        reliability_sorted = summary_df.sort_values("Thành công", ascending=False)
        for idx, row in reliability_sorted.iterrows():
            success_rate = row['Thành công'] / row['Tổng câu hỏi'] * 100
            print(f"   {idx+1}. {row['Model']}: {row['Thành công']}/{row['Tổng câu hỏi']} ({success_rate:.1f}%)")
        
        # Speed vs Reliability
        print("\n🎯 Speed vs Reliability:")
        for idx, row in summary_df.iterrows():
            success_rate = row['Thành công'] / row['Tổng câu hỏi'] * 100
            speed_score = 10 - (row['Thời gian TB (s)'] / summary_df['Thời gian TB (s)'].max() * 10)
            reliability_score = success_rate / 10
            overall_score = (speed_score + reliability_score) / 2
            print(f"   {row['Model']}: {overall_score:.1f}/10 (Speed: {speed_score:.1f}, Reliability: {reliability_score:.1f})")
    
    def save_results(self, summary_df: pd.DataFrame, detailed_df: pd.DataFrame):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"benchmark_summary_{self.dataset}_{timestamp}.csv"
        detailed_file = f"benchmark_detailed_{self.dataset}_{timestamp}.csv"
        
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Summary: {summary_file}")
        print(f"   📝 Detailed: {detailed_file}")
        
        return summary_file, detailed_file
    
    def plot_results(self, summary_df: pd.DataFrame):
        """Create visualization plots"""
        if not HAS_PLOTTING:
            print("\n⚠️ Plotting disabled (matplotlib not installed)")
            return
        
        print("\n📊 Generating plots...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Benchmark Results - {self.dataset.upper()}', fontsize=16, fontweight='bold')
        
        # 1. Average Response Time
        ax1 = axes[0, 0]
        summary_df.plot(x='Model', y='Thời gian TB (s)', kind='bar', ax=ax1, color='skyblue', legend=False)
        ax1.set_title('Average Response Time', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Success Rate
        ax2 = axes[0, 1]
        success_rate = (summary_df['Thành công'] / summary_df['Tổng câu hỏi'] * 100)
        ax2.bar(summary_df['Model'], success_rate, color='lightgreen')
        ax2.set_title('Success Rate', fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim([0, 105])
        
        # 3. Total Time
        ax3 = axes[1, 0]
        summary_df.plot(x='Model', y='Tổng thời gian (s)', kind='bar', ax=ax3, color='coral', legend=False)
        ax3.set_title('Total Time', fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Success vs Failure
        ax4 = axes[1, 1]
        x = range(len(summary_df))
        width = 0.35
        ax4.bar([i - width/2 for i in x], summary_df['Thành công'], width, label='Success', color='lightgreen')
        ax4.bar([i + width/2 for i in x], summary_df['Thất bại'], width, label='Failure', color='lightcoral')
        ax4.set_title('Success vs Failure', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.set_xlabel('')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df['Model'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"benchmark_plot_{self.dataset}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   📈 Plot saved: {plot_file}")
        
        # Show plot
        try:
            plt.show()
        except:
            print("   ℹ️ Cannot display plot (no display available)")
    
    def print_final_summary(self):
        """Print final summary"""
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETE")
        print("="*80)
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"📊 Dataset: {self.dataset}")
        print(f"🤖 Models tested: {len(self.models)}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Local Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 questions
  python benchmark.py --models qwen2.5 --dataset humaneval --num-questions 10
  
  # Full HumanEval benchmark
  python benchmark.py --models qwen2.5 gemma2:2b --dataset humaneval
  
  # Compare all models
  python benchmark.py --models qwen2.5 gemma2:2b llama3.1 mistral --dataset vimmrc --num-questions 50
  
  # Full dataset (no limit)
  python benchmark.py --models qwen2.5 --dataset vimmrc
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5"],
        help="Models to benchmark (default: qwen2.5)"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["vimmrc", "humaneval", "boolq", "squad", "commonsense"],
        default="humaneval",
        help="Dataset to use (default: humaneval)"
    )
    
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions (default: all)"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to CSV"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = BenchmarkRunner(args.models, args.dataset, args.num_questions)
    
    try:
        # Run benchmark
        summary_df, detailed_df, questions = runner.run_benchmark()
        
        # Print statistics
        runner.print_summary_stats(summary_df)
        runner.print_detailed_stats(summary_df, detailed_df)
        runner.print_comparison(summary_df)
        
        # Save results
        if not args.no_save:
            runner.save_results(summary_df, detailed_df)
        
        # Plot results
        if not args.no_plot and HAS_PLOTTING:
            runner.plot_results(summary_df)
        
        # Final summary
        runner.print_final_summary()
        
        print("\n🎉 Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
