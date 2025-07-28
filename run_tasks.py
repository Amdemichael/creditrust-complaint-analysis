#!/usr/bin/env python3
"""
Helper script to run the remaining tasks for the CrediTrust project.
This script will guide you through Task 3 and Task 4 completion.
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def run_command(command, description, cwd=None):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Error!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False
    
    return True

def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ Found: {filepath}")
        return True
    else:
        print(f"❌ Missing: {filepath}")
        return False

def main():
    """Main function to run all remaining tasks."""
    print_header("CrediTrust Complaint Analysis - Task Runner")
    
    print("This script will help you complete the remaining tasks:")
    print("• Task 3: RAG Core Logic and Evaluation")
    print("• Task 4: Interactive Chat Interface")
    
    # Check prerequisites
    print_step(1, "Checking Prerequisites")
    
    required_files = [
        "vector_store/faiss_index.bin",
        "vector_store/metadata.csv",
        "data/processed/filtered_complaints.csv"
    ]
    
    all_files_exist = True
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing required files. Please complete Task 1 and Task 2 first.")
        print("Run the notebooks in the notebooks/ directory.")
        return
    
    # Task 3: RAG Core Logic and Evaluation
    print_step(2, "Task 3: Testing RAG Pipeline")
    
    if not run_command("python src/test_rag.py", "Testing RAG pipeline"):
        print("\n❌ RAG pipeline test failed. Please check the error messages above.")
        return
    
    print_step(3, "Task 3: Running Evaluation")
    
    if not run_command("python src/evaluation.py", "Running evaluation framework"):
        print("\n❌ Evaluation failed. Please check the error messages above.")
        return
    
    # Check evaluation results
    if check_file_exists("reports/evaluation_results.csv"):
        print("\n✅ Evaluation completed successfully!")
        print("📊 Check reports/evaluation_results.csv for detailed results.")
    
    # Task 4: Interactive Chat Interface
    print_step(4, "Task 4: Launching Chat Interface")
    
    print("🚀 Starting Gradio interface...")
    print("The interface will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the server when done testing.")
    
    try:
        # Launch the interface
        subprocess.run("python app.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user.")
    
    # Final summary
    print_header("Task Completion Summary")
    
    print("✅ Completed Tasks:")
    print("  • Task 1: EDA and Preprocessing")
    print("  • Task 2: Chunking, Embedding, and Vector Store")
    print("  • Task 3: RAG Core Logic and Evaluation")
    print("  • Task 4: Interactive Chat Interface")
    
    print("\n📝 Next Steps for Final Submission:")
    print("1. Review evaluation results in reports/evaluation_results.csv")
    print("2. Take screenshots of the chat interface")
    print("3. Write your final report (Medium blog post format)")
    print("4. Prepare GitHub repository for submission")
    
    print("\n📊 Key Files Generated:")
    generated_files = [
        "src/rag_pipeline.py",
        "src/evaluation.py", 
        "src/test_rag.py",
        "app.py",
        "reports/evaluation_results.csv"
    ]
    
    for filepath in generated_files:
        if os.path.exists(filepath):
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath} (missing)")
    
    print("\n🎉 All tasks completed! Your RAG system is ready for submission.")

if __name__ == "__main__":
    main() 