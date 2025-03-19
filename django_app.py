# import os
# import sys
# import time
# import subprocess
# import re
# import webbrowser
# import requests

# def run_command(command):
#     """Runs a shell command and prints output in real-time."""
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
#     # Print command output in real-time
#     for line in process.stdout:
#         print(line, end="")  
    
#     stderr_output = process.stderr.read()
#     if process.wait() != 0:
#         print(f"❌ Error running command: {command}\n{stderr_output}")
#         sys.exit(1)

# def get_running_url():
#     """Finds the actual port Django is running on."""
#     time.sleep(2)  # Wait for logs to start
#     result = subprocess.run("netstat -ano | findstr LISTENING", shell=True, capture_output=True, text=True)
#     matches = re.findall(r"127.0.0.1:(\d+)", result.stdout)  # Look for listening ports on localhost
#     if matches:
#         return f"http://127.0.0.1:{matches[0]}"
#     return "http://127.0.0.1:8000"

# def wait_for_server(url, timeout=60*30):
#     """Waits until the Django server is fully running."""
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 print(f"✅ Server is ready at {url}")
#                 return
#         except requests.exceptions.ConnectionError:
#             pass  # Server is not yet ready
        
#         print("⏳ Waiting for Django server to start...")
#         time.sleep(10)

#     print("❌ Server did not start in time!")
#     sys.exit(1)

# def run_command_in_venv(command, venv_path):
#     """Runs a command inside the virtual environment."""
#     python_exec = os.path.join(venv_path, "Scripts", "python") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
#     full_command = f"{python_exec} -m {command}"
    
#     process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
#     # Print command output in real-time
#     for line in process.stdout:
#         print(line, end="")  
    
#     stderr_output = process.stderr.read()
#     if process.wait() != 0:
#         print(f"❌ Error running command: {command}\n{stderr_output}")
#         sys.exit(1)

# def main():
#     # Get current directory
#     current_dir = os.getcwd()
#     print("📁 Current Directory:", current_dir)

#     # Check if we are inside "deepfake-defender"
#     if os.path.basename(current_dir) == "deepfake-defender":
#         project_root = current_dir
#     else:
#         project_root = os.path.join(current_dir, "deepfake-defender")

#     project_dir = os.path.join(project_root, "deepfake")
#     venv_path = os.path.join(project_root, ".venv")

#     # Step 2: Create Virtual Environment if not exists
#     if not os.path.exists(venv_path):
#         print("🐍 Creating virtual environment...")
#         run_command(f"python -m venv {venv_path}")
#     else:
#         print("✅ Virtual environment already exists.")

#     # Step 3: Upgrade pip
#     print("⬆️ Upgrading pip...")
#     run_command_in_venv("pip install --upgrade pip", venv_path)

#     # Step 4: Install Project Dependencies
#     requirements_file = os.path.join(project_root, "requirements.txt")
#     if os.path.exists(requirements_file):
#         print("📦 Installing dependencies...")
#         run_command_in_venv(f"pip install -r {requirements_file}", venv_path)
#     else:
#         print("❌ requirements.txt not found!")
#         sys.exit(1)

#     # Step 5: Navigate to Project Directory
#     os.chdir(project_dir)
#     print(f"📂 Changed directory to: {project_dir}")

#     # Step 6: Start Django Development Server
#     print("🚀 Starting Django development server...")
#     python_exec = os.path.join(venv_path, "Scripts", "python") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
#     process = subprocess.Popen(f"{python_exec} manage.py runserver", shell=True)

#     # Step 7: Fetch the Correct URL
#     django_url = get_running_url()
    
#     # Step 8: Wait for the server to be fully ready
#     wait_for_server(django_url)

#     # Step 9: Open the browser only after the server is up
#     print(f"🌍 Opening browser at: {django_url}")
#     webbrowser.open(django_url)

#     print("✅ Django server is running. Press Ctrl+C to stop.")
#     process.wait()

# if __name__ == "__main__":
#     main()





import os
import sys
import time
import subprocess
import re
import webbrowser
import requests

def get_python_versions():
    """Find all installed Python versions from system paths and return as a sorted list (highest first)."""
    paths = os.getenv("PATH", "").split(os.pathsep)
    seen_paths = set()
    python_executables = []

    # Search for all possible Python executables in system paths
    for path in paths:
        if os.path.isdir(path) and path not in seen_paths:
            seen_paths.add(path)  # Avoid duplicates
            for file in os.listdir(path):
                if re.fullmatch(r"python(\d+(\.\d+)*)?\.exe", file, re.IGNORECASE):  # Match python.exe, python3.exe, python3.11.exe
                    full_path = os.path.realpath(os.path.join(path, file))
                    python_executables.append(full_path)

    versions = []
    for exe in python_executables:
        try:
            output = subprocess.run([exe, "--version"], capture_output=True, text=True)
            version_output = output.stdout.strip() or output.stderr.strip()
            match = re.search(r"Python (\d+\.\d+\.\d+)", version_output)
            if match:
                version_tuple = tuple(map(int, match.group(1).split(".")))  # Convert "3.11.2" -> (3, 11, 2)
                versions.append((exe, version_tuple))  # Store (path, version)
        except (FileNotFoundError, PermissionError):
            continue  # Skip invalid or inaccessible executables

    # Sort versions in descending order (highest version first)
    versions.sort(key=lambda x: x[1], reverse=True)

    return versions  # Returns list of tuples (python_path, version_tuple)

def find_compatible_python():
    """Find a Python version within the range 3.10 ≤ version ≤ 3.12.6"""
    versions = get_python_versions()
    
    print("\n🔍 Detected Python Versions:")
    for exe, v in versions:
        print(f"  ✅ {exe} - {'.'.join(map(str, v))}")

        
    for exe, v in versions:
        if (3, 10, 0) <= v <= (3, 12, 6):  # Check range
            print(f"\n🎯 Using Python: {exe} (Version {'.'.join(map(str, v))})")
            return exe  # Return the path to the correct Python executable

    return None  # No valid version found

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
    # time.sleep(2)  # Wait for logs to start
    # result = subprocess.run("netstat -ano | findstr LISTENING", shell=True, capture_output=True, text=True)
    # matches = re.findall(r"127.0.0.1:(\d+)", result.stdout)  # Look for listening ports on localhost
    # if matches:
    #     return f"http://127.0.0.1:{matches[0]}"
    # return "http://127.0.0.1:8000"

    # Giving a fixed url
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
    python_exec = find_compatible_python()
    
    if not python_exec:
        print("❌ No compatible Python version found (must be between 3.11 and 3.12.6).")
        print("🔗 Please install a supported version: https://www.python.org/downloads/")
        sys.exit(1)
    
    print(f"✅ Found compatible Python version: {python_exec}")

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
        run_command(f"{python_exec} -m venv {venv_path}")
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