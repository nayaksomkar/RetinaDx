"""
Create Virtual Environment
Run: python setup_venv.py
"""
import os
import subprocess
import sys

def create_venv():
    venv_name = "venv"
    
    # Check if already exists
    if os.path.exists(venv_name):
        print(f"'{venv_name}' already exists. Delete it manually to recreate.")
        return
    
    print(f"Creating virtual environment: {venv_name}")
    
    # Create venv
    subprocess.run([sys.executable, "-m", "venv", venv_name])
    
    # Get activation script path
    if os.name == "nt":  # Windows
        activate_script = os.path.join(venv_name, "Scripts", "activate.bat")
    else:  # Linux/Mac
        activate_script = os.path.join(venv_name, "bin", "activate")
    
    print(f"\nVirtual environment created!")
    print(f"\nTo activate (Windows):   {venv_name}\\Scripts\\activate")
    print(f"To activate (Linux/Mac): source {venv_name}/bin/activate")
    print(f"\nThen install dependencies:")
    print(f"  pip install -r requirements.txt")

if __name__ == "__main__":
    create_venv()