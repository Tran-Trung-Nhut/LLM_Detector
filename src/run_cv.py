import subprocess
from pathlib import Path

def main():
    for fold in range(5):
        split_path = f"data/splits/fold_{fold}.json"
        run_dir = f"runs/paligemma_single_image/fold_{fold}"
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "src.train_paligemma_lora_single_image",
        ]
        env = dict(**os.environ)
        env["SPLIT_PATH"] = split_path
        env["RUN_DIR"] = run_dir
        # nếu bạn sửa script để đọc env/args thì dùng được
        subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    main()