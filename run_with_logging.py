import sys
import os
from datetime import datetime
import subprocess
import threading

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create logs directory
    log_dir = os.path.join(script_dir, "prediction_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Get filename from user
    print(f"Default save location: {log_dir}")
    filename = input("Enter filename (without .txt, or press Enter for auto-name): ").strip()
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}"
    
    # Check if file exists
    output_path = os.path.join(log_dir, f"{filename}.txt")
    if os.path.exists(output_path):
        overwrite = input(f"File {filename}.txt already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Exiting. Please run again with a different name.")
            return
    
    print(f"\nSaving output to: {output_path}\n")
    print("Starting prediction...\n")
    print("=" * 50)
    print("INTERACTIVE MODE: You can now interact with the program")
    print("=" * 50)
    print()
    
    # Run main.py and capture output while allowing interaction
    python_path = os.path.join(script_dir, "venv", "Scripts", "python.exe")
    main_script = os.path.join(script_dir, "main.py")
    
    # Set UTF-8 encoding for Python
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'  # Disable buffering
    
    # Open the output file for writing
    output_file = open(output_path, 'w', encoding='utf-8')
    
    try:
        # Start the process with pipes for stdin, stdout, stderr
        process = subprocess.Popen(
            [python_path, main_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Thread to read output and save to file while also displaying
        def read_output():
            for line in process.stdout:
                # Write to file
                output_file.write(line)
                output_file.flush()
                # Display to console
                print(line, end='')
        
        # Start output reading thread
        output_thread = threading.Thread(target=read_output)
        output_thread.start()
        
        # Forward user input to the process
        try:
            while process.poll() is None:  # Check if process is still running
                user_input = sys.stdin.readline()
                if not user_input:
                    break
                if user_input.strip():  # Only send non-empty input
                    process.stdin.write(user_input)
                    process.stdin.flush()
                    output_file.write(user_input)
                    output_file.flush()
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        # Wait for process to complete
        process.wait()
        output_thread.join()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        output_file.close()
    
    print(f"\n✅ Complete! Output saved to: {output_path}")
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>> FIX ADDED HERE - This line prevents immediate closing <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("\nProgram finished. Press Enter to exit...")
    input()  # Waits for user to press Enter before closing
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")