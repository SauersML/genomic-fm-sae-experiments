import modal

app = modal.App("gpu-smoke")


@app.function(gpu="A100-80GB", image=modal.Image.debian_slim())
def check():
    import subprocess
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        capture_output=True, text=True).stdout)


@app.local_entrypoint()
def main():
    check.remote()
