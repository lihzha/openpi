# download_wandb_images.py
import os

import wandb

ENTITY = "lihanzha"
PROJECT = "openpi"
RUN_ID = "sfq2189i"  # from your URL
OUTDIR = "wandb_panel_images"

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

os.makedirs(OUTDIR, exist_ok=True)

# List all files in the run; filter to images logged as Media
for f in run.files(per_page=10000):
    # Typical image paths look like: "media/images/<key>_<step>_...png"
    if f.name.startswith("media/images/"):
        print("Downloading:", f.name)
        f.download(root=OUTDIR, replace=True)

print(f"Done. Images saved under ./{OUTDIR}")
