"""Download the pretrained weights and tokenizer used by the experiment jobs."""

import argparse
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VILT_URL = "https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models", help="Destination for all model assets.")
    parser.add_argument("--models", choices=["vision", "vilt", "clip", "all"], default="all")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(root / "huggingface" / "hub")
    import torch
    torch.hub.set_dir(str(root / "torch"))
    if args.models in ("vision", "all"):
        import torchvision
        import timm
        weights = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
        weights.get_state_dict(progress=True, check_hash=True)
        for name in ("deit_base_patch16_224.fb_in1k", "convit_base.fb_in1k"):
            model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="")
            del model
        print(f"Vision weights downloaded into {root}")
    if args.models in ("vilt", "all"):
        checkpoint = root / "vilt_200k_mlm_itm.ckpt"
        if not checkpoint.exists():
            temporary = checkpoint.with_suffix(".ckpt.part")
            urllib.request.urlretrieve(VILT_URL, temporary)
            temporary.replace(checkpoint)
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tokenizer.save_pretrained(root / "bert-base-uncased")
        print(f"ViLT weights and BERT tokenizer downloaded into {root}")
    if args.models in ('clip', 'all'):
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        name = 'openai/clip-vit-base-patch32'
        destination = root / 'clip-vit-base-patch32'
        CLIPModel.from_pretrained(name).save_pretrained(destination)
        CLIPTokenizer.from_pretrained(name).save_pretrained(destination)
        CLIPImageProcessor.from_pretrained(name).save_pretrained(destination)
        print(f'CLIP weights, tokenizer, and image processor downloaded into {destination}')


if __name__ == "__main__":
    main()
