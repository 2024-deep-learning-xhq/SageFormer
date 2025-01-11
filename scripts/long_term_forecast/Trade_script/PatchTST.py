import os
import subprocess
from itertools import product


def run_experiments():
    # 定义要搜索的超参数空间
    param_grid = {
        "pred_len": [96],
        "n_heads": [4, 8, 16, 32],
        "e_layers": [2],
        "d_model": [32, 64, 128, 256],
    }

    # 基础配置
    base_config = {
        "seq_len": 192,
        "model_name": "PatchTST",
        "root_path_name": "./dataset/weather/",
        "data_path_name": "5min.csv",
        "model_id_name": "5min",
        "data_name": "custom",
        "random_seed": 2021,
    }

    # 创建日志目录
    os.makedirs("logs/LongForecasting", exist_ok=True)

    # 生成并执行所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_experiments = len(list(product(*param_values)))
    print(f"Total experiments to run: {total_experiments}")

    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        config = base_config.copy()
        config.update(params)

        # 构建命令
        cmd = [
            "python",
            "-u",
            "run_TST.py",
            "--random_seed",
            str(config["random_seed"]),
            "--seasonal_patterns",
            "Monthly",
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--root_path",
            config["root_path_name"],
            "--data_path",
            config["data_path_name"],
            "--model_id",
            f"{config['model_id_name']}_{config['seq_len']}_{params['pred_len']}",
            "--model",
            config["model_name"],
            "--data",
            config["data_name"],
            "--features",
            "M",
            "--seq_len",
            str(config["seq_len"]),
            "--pred_len",
            str(params["pred_len"]),
            "--enc_in",
            "22",
            "--e_layers",
            str(params["e_layers"]),
            "--n_heads",
            str(params["n_heads"]),
            "--d_model",
            str(params["d_model"]),
            "--d_ff",
            "256",
            "--dropout",
            "0.2",
            "--fc_dropout",
            "0.2",
            "--head_dropout",
            "0",
            "--patch_len",
            "16",
            "--stride",
            "8",
            "--des",
            "Exp",
            "--train_epochs",
            "100",
            "--patience",
            "10",
            "--lradj",
            "type3",
            "--pct_start",
            "0.2",
            "--itr",
            "1",
            "--batch_size",
            "16",
            "--learning_rate",
            "0.0001",
        ]

        # 构建日志文件名
        log_file = f"logs/LongForecasting/{config['model_name']}_{config['model_id_name']}_{config['seq_len']}_{params['pred_len']}_h{params['n_heads']}_l{params['e_layers']}_d{params['d_model']}.log"

        # 打印当前实验信息
        print(f"\nRunning experiment {i}/{total_experiments}")
        print(
            f"Parameters: pred_len={params['pred_len']}, n_heads={params['n_heads']}, e_layers={params['e_layers']}, d_model={params['d_model']}"
        )
        print(f"Log file: {log_file}")

        # 执行命令并将输出重定向到日志文件
        with open(log_file, "w") as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if process.returncode == 0:
            print(f"Experiment {i} completed successfully")
        else:
            print(f"Experiment {i} failed with return code {process.returncode}")


if __name__ == "__main__":
    run_experiments()
