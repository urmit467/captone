"""Week 4a: appearance embedding per track from the crops saved in Week 3.

  python extract_embeddings.py --crops ../week3_detection_tracking/output/crops --backend hist
  python extract_embeddings.py --crops ... --backend osnet --weights osnet_x1_0_market.pth --device cuda:0

Backends
  hist      HSV colour histograms of torso / legs. No deep model needed: use it as the baseline, or as the fallback
            when participants wear distinctly coloured clothes.
  osnet     OSNet (torchreid) - the recommended Re-ID model. Needs a pretrained weights file (Market-1501 / MSMT17 from the
            torchreid model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO ). Pass with --weights.
  resnet50  ImageNet ResNet-50 features (weak generic baseline, downloads weights from torchvision).
Output npz: keys = 'camera_id|track_id', emb = L2-normalised mean embedding per track.
"""
import argparse, glob, os
import cv2, numpy as np


class HistEmbedder:
    dim = 16 * 8 * 2 + 4 * 2

    def _one(self, bgr):
        img = cv2.resize(bgr, (64, 128))[:, 10:54]                       # centre crop: less background
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feats = []
        for a, b in ((int(.15 * 128), int(.50 * 128)), (int(.50 * 128), 128)):   # torso, legs
            s = hsv[a:b]
            hs = cv2.calcHist([s], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
            v = cv2.calcHist([s], [2], None, [4], [0, 256]).ravel()
            feats += [np.sqrt(hs / max(hs.sum(), 1)), np.sqrt(v / max(v.sum(), 1))]
        return np.concatenate(feats).astype(np.float32)

    def __call__(self, imgs):
        return np.stack([self._one(i) for i in imgs])


class OsnetEmbedder:
    def __init__(self, weights, device):
        try:
            from torchreid.utils import FeatureExtractor          # GitHub install of deep-person-reid
        except ImportError:
            from torchreid.reid.utils import FeatureExtractor      # PyPI 'torchreid' package layout
        self.fx = FeatureExtractor(model_name="osnet_x1_0", model_path=weights, device=device or "cpu", image_size=(256, 128))

    def __call__(self, imgs):
        rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        return self.fx(rgb).cpu().numpy()


class ResnetEmbedder:
    def __init__(self, device):
        import torch, torchvision
        self.torch, self.dev = torch, device or "cpu"
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        self.m = m.eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, imgs):
        t = self.torch
        x = np.stack([cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), (128, 256)) for i in imgs])
        x = (t.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        with t.no_grad():
            return self.m(x.to(self.dev)).cpu().numpy()


def make_embedder(backend, weights=None, device=None):
    if backend == "hist":
        return HistEmbedder()
    if backend == "osnet":
        if not weights:
            raise SystemExit("--weights is required for the osnet backend (see docstring)")
        return OsnetEmbedder(weights, device)
    if backend == "resnet50":
        return ResnetEmbedder(device)
    raise SystemExit(f"unknown backend {backend}")


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def track_crops(crops_dir):
    """yield (camera_id, track_id, [image paths])"""
    for cam_dir in sorted(glob.glob(os.path.join(crops_dir, "*"))):
        for tdir in sorted(glob.glob(os.path.join(cam_dir, "*"))):
            files = sorted(glob.glob(os.path.join(tdir, "*.jpg")))
            if files:
                yield os.path.basename(cam_dir), int(os.path.basename(tdir)), files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--backend", default="hist", choices=["hist", "osnet", "resnet50"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="track_embeddings.npz")
    a = ap.parse_args()
    emb_fn = make_embedder(a.backend, a.weights, a.device)
    keys, embs = [], []
    for cam, tid, files in track_crops(a.crops):
        imgs = [im for im in (cv2.imread(f) for f in files) if im is not None]
        e = l2(emb_fn(imgs)).mean(0)
        keys.append(f"{cam}|{tid}"); embs.append(l2(e))
    np.savez(a.out, keys=np.array(keys), emb=np.stack(embs))
    print(f"{len(keys)} track embeddings ({a.backend}, dim {embs[0].shape[0]}) -> {a.out}")


if __name__ == "__main__":
    main()
