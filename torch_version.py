import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU")

    # sh scripts/long_term_forecast/Weather_script/SageFormer.sh