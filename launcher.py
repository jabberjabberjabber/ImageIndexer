#!/usr/bin/env -S uv run
"""
ImageIndexer Launcher
Cross-platform launcher for the ImageIndexer application.
"""
import subprocess
import sys
import json
import os
import platform
import runpy
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    if platform.system() == "Windows":
        # Try to enable ANSI colors on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    GRAY = '\033[0;37m'
    WHITE = '\033[1;37m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR / "resources"
KOBOLD_ARGS_PATH = RESOURCES_DIR / "kobold_args.json"

# Track koboldcpp process for cleanup only
_kobold_process = None


def check_dependencies():
    """Check if required dependencies are installed."""
    # Check for exiftool
    try:
        subprocess.run(["exiftool", "-ver"], capture_output=True, check=True)
        print(f"{Colors.GREEN}exiftool is installed.{Colors.NC}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.YELLOW}exiftool is not installed.{Colors.NC}")
        return install_exiftool()


def install_exiftool():
    """Attempt to install exiftool based on platform."""
    print(f"{Colors.CYAN}Installing exiftool...{Colors.NC}")

    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["winget", "install", "-e", "--id", "OliverBetz.ExifTool"], check=True)
        elif system == "Darwin":
            subprocess.run(["brew", "install", "exiftool"], check=True)
        elif system == "Linux":
            # Try different package managers
            for pm_cmd in [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "libimage-exiftool-perl"]
            ]:
                try:
                    subprocess.run(pm_cmd, check=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

        print(f"{Colors.GREEN}exiftool installed successfully.{Colors.NC}")
        print(f"{Colors.YELLOW}Please restart the launcher.{Colors.NC}")
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.RED}Failed to install exiftool automatically.{Colors.NC}")
        print(f"{Colors.YELLOW}Please install exiftool manually.{Colors.NC}")
        return False


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def show_menu():
    """Display the main menu."""
    clear_screen()
    print(f"{Colors.CYAN}================ Indexer Launcher ================{Colors.NC}")
    print()
    print(f"{Colors.YELLOW}1:{Colors.NC} {Colors.GREEN}Install Requirements{Colors.NC}")
    print(f"{Colors.YELLOW}2:{Colors.NC} {Colors.GREEN}Launch Model (koboldcpp){Colors.NC}")
    print(f"{Colors.YELLOW}3:{Colors.NC} {Colors.GREEN}Launch Indexer GUI{Colors.NC}")
    print(f"{Colors.YELLOW}4:{Colors.NC} {Colors.GREEN}Select Model{Colors.NC}")
    print(f"{Colors.YELLOW}Q:{Colors.NC} {Colors.RED}Quit{Colors.NC}")
    print()


def run_setup():
    """Run the setup/installation."""
    print(f"{Colors.BLUE}Running setup...{Colors.NC}")

    if not check_dependencies():
        input("Press Enter to continue...")
        return

    print(f"{Colors.CYAN}Installing Python dependencies with uv...{Colors.NC}")
    try:
        subprocess.run(["uv", "sync"], cwd=SCRIPT_DIR, check=True)
        print(f"{Colors.GREEN}Dependencies installed successfully.{Colors.NC}")
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}Failed to install dependencies.{Colors.NC}")
        input("Press Enter to continue...")
        return

    clear_screen()
    print()
    print("******************************************************")
    print("** AFTER SELECTING A MODEL AN EXIT CODE WILL APPEAR **")
    print("**              THIS IS NOT AN ERROR                **")
    print("**        CLOSE THIS WINDOW WHEN YOU ARE DONE       **")
    print("******************************************************")
    print()

    # Run setup module directly
    sys.argv = ["llmii_setup", "--update"]
    try:
        runpy.run_module("src.llmii_setup", run_name="__main__")
    except SystemExit:
        pass  # Module may call sys.exit()

    input("Press Enter to continue...")


def run_gui():
    """Launch the GUI component."""
    print(f"{Colors.CYAN}Launching GUI...{Colors.NC}")
    # Run GUI in subprocess to avoid Qt reinitialization issues
    # Qt doesn't like being started multiple times in the same process
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.llmii_gui"],
            cwd=SCRIPT_DIR
        )
    except Exception as e:
        print(f"{Colors.RED}GUI failed to launch: {e}{Colors.NC}")
        input("Press Enter to continue...")


def launch_model():
    """Launch koboldcpp in a new terminal window."""
    global _kobold_process

    if not KOBOLD_ARGS_PATH.exists():
        print(f"{Colors.RED}Error: Model configuration not found.{Colors.NC}")
        print(f"{Colors.YELLOW}Please run 'Select Model' first.{Colors.NC}")
        input("Press Enter to continue...")
        return

    print(f"{Colors.CYAN}Loading model configuration...{Colors.NC}")

    try:
        with open(KOBOLD_ARGS_PATH, 'r') as f:
            config = json.load(f)

        exe_path = RESOURCES_DIR / config['executable']
        working_dir = exe_path.parent

        # Build command
        cmd = [
            str(exe_path),
            config['model_param'],
            "--mmproj", config['mmproj'],
            "--contextsize", str(config['contextsize']),
            "--visionmaxres", str(config['visionmaxres']),
            "--chatcompletionsadapter", config['chatcompletionsadapter']
        ]

        if config.get('flashattention', False):
            cmd.append("--flashattention")

        print(f"{Colors.GREEN}Launching koboldcpp...{Colors.NC}")

        # Make executable on Unix
        if platform.system() != "Windows":
            os.chmod(exe_path, 0o755)

        # Launch in new window based on platform
        system = platform.system()
        if system == "Windows":
            _kobold_process = subprocess.Popen(cmd, cwd=working_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif system == "Darwin":
            # macOS
            cmd_str = ' '.join([f'"{arg}"' for arg in cmd])
            applescript = f'tell application "Terminal" to do script "cd \\"{working_dir}\\" && {cmd_str}"'
            subprocess.run(["osascript", "-e", applescript])
            # Save exe name for cleanup
            _kobold_process = exe_path.name
        else:
            # Linux - use $TERMINAL or fallback
            terminal = os.environ.get('TERMINAL', 'x-terminal-emulator')
            cmd_str = ' '.join([f'"{arg}"' for arg in cmd])
            try:
                subprocess.Popen([terminal, "-e", f"bash -c 'cd \"{working_dir}\" && {cmd_str}; exec bash'"])
                # Save exe name for cleanup
                _kobold_process = exe_path.name
            except FileNotFoundError:
                print(f"{Colors.YELLOW}No terminal found. Set $TERMINAL env var.{Colors.NC}")
                input("Press Enter to continue...")
                return

        print(f"{Colors.GREEN}Model launched in new window!{Colors.NC}")
        time.sleep(1)

    except Exception as e:
        print(f"{Colors.RED}Error launching model: {e}{Colors.NC}")
        input("Press Enter to continue...")


def select_model():
    """Run model selection."""
    print(f"{Colors.BLUE}Model selection starting...{Colors.NC}")
    print()
    print("******************************************************")
    print("** AFTER SELECTING A MODEL AN EXIT CODE WILL APPEAR **")
    print("**                                                  **")
    print("**              THIS IS NOT AN ERROR                **")
    print("******************************************************")
    print()

    # Run setup module directly (without --update flag)
    sys.argv = ["llmii_setup"]
    try:
        runpy.run_module("src.llmii_setup", run_name="__main__")
    except SystemExit:
        pass  # Module may call sys.exit()

    input("Press Enter to continue...")


def cleanup():
    """Cleanup function to kill koboldcpp on exit."""
    global _kobold_process

    if not _kobold_process:
        return

    print(f"\n{Colors.YELLOW}Cleaning up koboldcpp...{Colors.NC}")

    # Windows: we have a process handle
    if isinstance(_kobold_process, subprocess.Popen):
        if _kobold_process.poll() is None:
            _kobold_process.terminate()
            try:
                _kobold_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kobold_process.kill()
            print(f"{Colors.GREEN}koboldcpp terminated.{Colors.NC}")
    else:
        # Unix: we have a process name, use pkill
        try:
            subprocess.run(["pkill", "-f", _kobold_process], check=False)
            print(f"{Colors.GREEN}koboldcpp terminated.{Colors.NC}")
        except:
            pass


def main():
    """Main entry point."""
    import atexit
    atexit.register(cleanup)

    selection = ""
    while selection.lower() != 'q':
        show_menu()
        selection = input(f"{Colors.CYAN}Please make a selection:{Colors.NC} ").strip()

        if selection == '1':
            run_setup()
        elif selection == '2':
            launch_model()
        elif selection == '3':
            run_gui()
        elif selection == '4':
            select_model()
        elif selection.lower() == 'q':
            print(f"{Colors.MAGENTA}Exiting...{Colors.NC}")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.MAGENTA}Interrupted. Exiting...{Colors.NC}")
        cleanup()
        sys.exit(0)
