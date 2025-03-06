import os
import sys
import time
import subprocess
import re
import webbrowser
import requests

def run_command(command):
    """Runs a shell command and prints output in real-time."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print command output in real-time
    for line in process.stdout:
        print(line, end="")  
    
    stderr_output = process.stderr.read()
    if process.wait() != 0:
        print(f"❌ Error running command: {command}\n{stderr_output}")
        sys.exit(1)

def get_running_url():
    """Finds the actual port Django is running on."""
    time.sleep(2)  # Wait for logs to start
    result = subprocess.run("netstat -ano | findstr LISTENING", shell=True, capture_output=True, text=True)
    matches = re.findall(r"127.0.0.1:(\d+)", result.stdout)  # Look for listening ports on localhost
    if matches:
        return f"http://127.0.0.1:{matches[0]}"
    return "http://127.0.0.1:8000"

def wait_for_server(url, timeout=60*30):
    """Waits until the Django server is fully running."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"✅ Server is ready at {url}")
                return
        except requests.exceptions.ConnectionError:
            pass  # Server is not yet ready
        
        print("⏳ Waiting for Django server to start...")
        time.sleep(10)

    print("❌ Server did not start in time!")
    sys.exit(1)

def run_command_in_venv(command, venv_path):
    """Runs a command inside the virtual environment."""
    python_exec = os.path.join(venv_path, "Scripts", "python") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    full_command = f"{python_exec} -m {command}"
    
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print command output in real-time
    for line in process.stdout:
        print(line, end="")  
    
    stderr_output = process.stderr.read()
    if process.wait() != 0:
        print(f"❌ Error running command: {command}\n{stderr_output}")
        sys.exit(1)

def main():
    # Get current directory
    current_dir = os.getcwd()
    print("📁 Current Directory:", current_dir)

    # Check if we are inside "deepfake-defender"
    if os.path.basename(current_dir) == "deepfake-defender":
        project_root = current_dir
    else:
        project_root = os.path.join(current_dir, "deepfake-defender")

    project_dir = os.path.join(project_root, "deepfake")
    venv_path = os.path.join(project_root, ".venv")

    # Step 2: Create Virtual Environment if not exists
    if not os.path.exists(venv_path):
        print("🐍 Creating virtual environment...")
        run_command(f"python -m venv {venv_path}")
    else:
        print("✅ Virtual environment already exists.")

    # Step 3: Upgrade pip
    print("⬆️ Upgrading pip...")
    run_command_in_venv("pip install --upgrade pip", venv_path)

    # Step 4: Install Project Dependencies
    requirements_file = os.path.join(project_root, "requirements.txt")
    if os.path.exists(requirements_file):
        print("📦 Installing dependencies...")
        run_command_in_venv(f"pip install -r {requirements_file}", venv_path)
    else:
        print("❌ requirements.txt not found!")
        sys.exit(1)

    # Step 5: Navigate to Project Directory
    os.chdir(project_dir)
    print(f"📂 Changed directory to: {project_dir}")

    # Step 6: Start Django Development Server
    print("🚀 Starting Django development server...")
    python_exec = os.path.join(venv_path, "Scripts", "python") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    process = subprocess.Popen(f"{python_exec} manage.py runserver", shell=True)

    # Step 7: Fetch the Correct URL
    django_url = get_running_url()
    
    # Step 8: Wait for the server to be fully ready
    wait_for_server(django_url)

    # Step 9: Open the browser only after the server is up
    print(f"🌍 Opening browser at: {django_url}")
    webbrowser.open(django_url)

    print("✅ Django server is running. Press Ctrl+C to stop.")
    process.wait()

if __name__ == "__main__":
    main()
