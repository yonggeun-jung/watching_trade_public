import subprocess

STEPS = [
    "watching_trade/src_russia/01_ports.py",
    "watching_trade/src_russia/02_sar.py",
    "watching_trade/src_russia/03_viirs.py",
    "watching_trade/src_russia/04_merge.py",
    "watching_trade/src_russia/05_predict.py"]


if __name__ == "__main__":
    print("\nStarting execution...\n")
    
    for step in STEPS:

        process = subprocess.Popen(
            ["python3", step],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        for line in process.stderr:
            print(line, end="")

        returncode = process.wait()
        if returncode == 0:
            print(f"--- Finished {step} ---")
        else:
            print(f"*** ERROR in {step}, return code {returncode} ***")
            print("Execution aborted.")
            raise SystemExit(returncode)

    print("\nExecution completed.\n")

# CHECKED