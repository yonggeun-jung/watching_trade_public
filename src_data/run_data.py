import subprocess

STEPS = [
    "watching_trade/src_data/01_ports.py",
    "watching_trade/src_data/02_sar.py",
    "watching_trade/src_data/03_viirs.py",
    "watching_trade/src_data/04_trade.py",
    "watching_trade/src_data/05_merge.py"]


if __name__ == "__main__":
    print("\nStarting data extraction...\n")
    
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
            print("Data extraction aborted.")
            raise SystemExit(returncode)

    print("\nData extraction completed.\n")

# CHECKED.