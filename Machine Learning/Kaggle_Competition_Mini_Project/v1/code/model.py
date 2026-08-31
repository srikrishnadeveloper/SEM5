# model.py - segmentation model and EMA helpers
import copy
import torch
import segmentation_models_pytorch as smp
import config


def get_model(encoder_name=config.ENCODER, model_name=config.MODEL_NAME,
              in_channels=config.IN_CHANNELS, classes=config.CLASSES,
              pretrained=config.PRETRAINED):
    names = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
             "deeplabv3plus": smp.DeepLabV3Plus, "fpn": smp.FPN}
    key = model_name.lower()
    if key not in names:
        raise ValueError(f"unknown model_name {model_name}; choose {sorted(names)}")
    kwargs = dict(encoder_name=encoder_name,
                  encoder_weights="imagenet" if pretrained else None,
                  in_channels=in_channels, classes=classes, activation=None)
    if key in {"unet", "unetplusplus", "fpn"}:
        kwargs["decoder_use_batchnorm"] = config.DECODER_USE_BATCHNORM
    try:
        net = names[key](**kwargs)
    except (TypeError, ValueError):
        # inplace-abn may be unavailable, or an encoder may have no ImageNet weights.
        kwargs["decoder_use_batchnorm"] = True
        try:
            net = names[key](**kwargs)
        except Exception:
            kwargs["encoder_weights"] = None
            net = names[key](**kwargs)
    if config.GRADIENT_CHECKPOINTING and hasattr(net.encoder, "set_grad_checkpointing"):
        net.encoder.set_grad_checkpointing(True)
    return net


def create_ema_model(net, decay=config.EMA_DECAY):
    """Return a detached model copy with an update_parameters(model) method."""
    ema = copy.deepcopy(net).eval()
    ema.requires_grad_(False)
    ema.decay = decay
    @torch.no_grad()
    def update_parameters(source):
        for ep, p in zip(ema.parameters(), source.parameters()):
            ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for eb, b in zip(ema.buffers(), source.buffers()):
            eb.copy_(b)
    ema.update_parameters = update_parameters
    return ema


def load_checkpoint(net, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    weights = state.get("ema_model") or state.get("model") or state
    net.load_state_dict(weights)
    print(f"loaded checkpoint from {checkpoint_path}")
    return net


if __name__ == "__main__":
    m = get_model(pretrained=False)
    x = torch.randn(1, config.IN_CHANNELS, min(256, config.TRAIN_RES), min(256, config.TRAIN_RES))
    print(f"input shape: {x.shape}  output shape: {m(x).shape}")
