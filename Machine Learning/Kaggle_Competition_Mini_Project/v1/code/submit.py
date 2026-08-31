# submit.py - OOF/test submission CLI
import argparse, glob, os
import config, infer


def main():
    p = argparse.ArgumentParser(description="generate filament submission csv")
    p.add_argument("--checkpoint"); p.add_argument("--out")
    p.add_argument("--image-size", type=int, default=config.TRAIN_RES)
    p.add_argument("--threshold", type=float, default=config.PROB_THRESHOLD)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--val-fold", type=int)
    p.add_argument("--test", action="store_true", help="explicitly run test inference")
    p.add_argument("--ensemble-dir", action="append", default=[], help="probability directory; repeat or pass a parent containing fold dirs")
    a = p.parse_args(); dirs = []
    for d in a.ensemble_dir:
        children = [x for x in glob.glob(os.path.join(d, "*")) if os.path.isdir(x)]
        dirs.extend(children if children and not glob.glob(os.path.join(d, "*.npy")) else [d])
    if a.val_fold is not None and not a.test:
        out = infer.run_validation_oof(a.val_fold, a.image_size, a.threshold, a.tta, dirs or None)
    else:
        out = infer.run_test_submission(a.checkpoint, a.out, a.image_size, a.threshold, a.tta, dirs or None)
    print("wrote", out if isinstance(out, str) else a.out or config.SUBMISSIONS_DIR)


if __name__ == "__main__": main()
