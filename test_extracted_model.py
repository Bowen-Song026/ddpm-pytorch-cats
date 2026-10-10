from model_extracted import Unet, GaussianDiffusion
from ema_pytorch import EMA
import torch
from torchvision import utils
from pathlib import Path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False).to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=50,
        objective='pred_v',
        ddim_sampling_eta=0.0
    ).to(device)

    ema = EMA(model, beta=0.995).to(device)

    results_folder = Path('./mixed_cat_results')

    ema_path = results_folder / 'ema-final.pt'
    if ema_path.exists():
        print(f"Loading EMA weights from {ema_path}")
        ema.load_state_dict(torch.load(ema_path, map_location=device))
    else:
        print(f"Loading model weights")
        model.load_state_dict(torch.load(results_folder / 'model-final.pt', map_location=device))
        ema = model

    ema.eval()

    print("Generating test image...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            sample = diffusion.sample(batch_size=4)
        utils.save_image((sample+1)*0.5, str(results_folder / 'extracted_model_test.png'), nrow=2)

    print(f"Done! Test image saved to {results_folder / 'extracted_model_test.png'}")

if __name__ == '__main__':
    main()