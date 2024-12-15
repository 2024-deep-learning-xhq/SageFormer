import os
import subprocess


def run_experiment():
    # Define constants
    seq_len = 192
    model_name = "PatchTST"
    root_path_name = "./dataset/weather/"
    data_path_name = "5min.csv"
    model_id_name = "5min"
    data_name = "custom"
    random_seed = 2021

    # Ensure logs directory exists
    os.makedirs("logs/LongForecasting", exist_ok=True)

    # Loop through pred_len values (in this case only one value: 96)
    for pred_len in [96]:
        # Construct the model_id
        model_id = f"{model_id_name}_{seq_len}_{pred_len}"

        # Construct the command
        cmd = [
            "python",
            "-u",
            "run_TST.py",
            "--random_seed",
            str(random_seed),
            "--seasonal_patterns",
            "Monthly",
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--root_path",
            root_path_name,
            "--data_path",
            data_path_name,
            "--model_id",
            model_id,
            "--model",
            model_name,
            "--data",
            data_name,
            "--features",
            "M",
            "--seq_len",
            str(seq_len),
            "--pred_len",
            str(pred_len),
            "--enc_in",
            "22",
            "--e_layers",
            "3",
            "--n_heads",
            "16",
            "--d_model",
            "128",
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

        # Construct the log file path
        log_file = f"logs/LongForecasting/{model_name}_{model_id_name}_{seq_len}_{pred_len}.log"

        # Run the command and redirect output to log file
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    run_experiment()
