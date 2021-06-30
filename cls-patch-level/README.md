# Train on _~20K_ annotations from 138 slides
> Using classification task for the feature extraction.
> Detection model can also test in this classification experience.

## One State-Of-The-Art (SOTA) model:
- MobileNetV2 (MNV2) <- `torchvision.mobilenet.MoobileNetV2`

## The proposed InCNet adopted into MNV2 architecture:
- Inline Connecting Network (InCNet) <- `cls_icn.py`

## Detection model
- The proposed model build on InCNet into YOLOv3-Tiny architecture (YOLCO) <- `det_models.py`
- The `bbox loss` in YOLCO removed when training (YOCO) <- `det_models.py`

## Notes
- `slide_readtool/` includes the private WSI format used in the experience, please download from link:
- `data/` includes the private bbox annotations used in the experience, please download from link:
