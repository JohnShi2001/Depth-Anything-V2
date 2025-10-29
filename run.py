from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.depthpro_fusion import (
    DepthProDistanceEstimator,
    DepthProFusion,
    TractorScenarioEvaluator,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str, required=True, help='Image or directory path')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save raw DAV2 depth (float32 npy)')
    parser.add_argument('--save-absolute', dest='save_absolute', action='store_true', help='save fused absolute depth (float32 npy)')

    parser.add_argument('--fusion-strategy', type=str, default='scale_shift', choices=['none', 'scale_shift', 'hybrid'], help='Depth fusion strategy')
    parser.add_argument('--fusion-hybrid-weight', type=float, default=0.5, help='Blend weight for hybrid fusion (DepthPro vs DAV2)')
    parser.add_argument('--depthpro-checkpoint', type=str, default=None, help='TorchScript checkpoint for DepthPro scale/shift estimator')
    parser.add_argument('--depthpro-metadata-dir', type=str, default=None, help='Directory with DepthPro per-image scale/shift metadata')
    parser.add_argument('--depthpro-default-scale', type=float, default=1.0, help='Fallback scale when DepthPro data missing')
    parser.add_argument('--depthpro-default-shift', type=float, default=0.0, help='Fallback shift when DepthPro data missing')
    parser.add_argument('--depthpro-device', type=str, default=None, help='Device for DepthPro checkpoint (default: align with DAV2)')

    parser.add_argument('--max-depth', type=float, default=None, help='Clip fused depth to this maximum value (meters)')
    parser.add_argument('--tractor-gt-dir', type=str, default=None, help='Directory with tractor scenario ground truth depth maps')
    parser.add_argument('--tractor-report', type=str, default=None, help='Path to save aggregated tractor metrics JSON report')
    parser.add_argument('--auto-tune-fusion', dest='auto_tune_fusion', action='store_true', help='Fit linear scale/shift against ground truth')
    parser.add_argument('--tractor-percentile', type=float, default=99.0, help='Percentile used to suggest dynamic max depth from tractor GT')

    return parser.parse_args()


def build_file_list(path: str) -> List[str]:
    if os.path.isfile(path):
        if path.endswith('txt'):
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return [path]
    files = [p for p in glob.glob(os.path.join(path, '**/*'), recursive=True) if os.path.isfile(p)]
    return files


def create_estimator(args: argparse.Namespace, device: str) -> Optional[DepthProDistanceEstimator]:
    if args.fusion_strategy == 'none':
        return None

    depthpro_device = args.depthpro_device or device
    return DepthProDistanceEstimator(
        checkpoint_path=args.depthpro_checkpoint,
        metadata_dir=args.depthpro_metadata_dir,
        default_scale=args.depthpro_default_scale,
        default_shift=args.depthpro_default_shift,
        device=depthpro_device,
    )


def prepare_model(args: argparse.Namespace, device: str) -> DepthAnythingV2:
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything


def normalise_depth(depth_map: np.ndarray) -> np.ndarray:
    depth_min = float(np.min(depth_map))
    depth_max = float(np.max(depth_map))
    if depth_max - depth_min > 1e-6:
        return (depth_map - depth_min) / (depth_max - depth_min)
    return np.zeros_like(depth_map)


def colourise(depth_map: np.ndarray, cmap) -> np.ndarray:
    normalised = normalise_depth(depth_map)
    coloured = cmap(normalised)[..., :3] * 255.0
    return coloured.astype(np.uint8)[:, :, ::-1]


def main():
    args = parse_arguments()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    depth_anything = prepare_model(args, DEVICE)

    filenames = build_file_list(args.img_path)
    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    depthpro_estimator = create_estimator(args, DEVICE)
    clip_value = args.max_depth

    evaluator = None
    if args.tractor_gt_dir:
        evaluator = TractorScenarioEvaluator(
            gt_dir=args.tractor_gt_dir,
            report_path=args.tractor_report,
            percentile=args.tractor_percentile,
        )

    for index, filename in enumerate(filenames, start=1):
        print(f'Progress {index}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print('  Warning: failed to load image, skipping')
            continue

        raw_depth = depth_anything.infer_image(raw_image, args.input_size)
        basename = os.path.splitext(os.path.basename(filename))[0]

        if args.save_numpy:
            np.save(os.path.join(args.outdir, f'{basename}_raw_depth.npy'), raw_depth)

        fused_depth = raw_depth.astype(np.float32)

        if args.fusion_strategy != 'none':
            if depthpro_estimator is None:
                depthpro_estimator = create_estimator(args, DEVICE)
            if depthpro_estimator is not None:
                scale_shift = depthpro_estimator.estimate(raw_image, basename)
                fused_depth = DepthProFusion.apply(
                    raw_depth,
                    scale_shift,
                    strategy=args.fusion_strategy,
                    hybrid_weight=args.fusion_hybrid_weight,
                )

        if evaluator is not None:
            if args.auto_tune_fusion:
                tuned = evaluator.solve_linear_regression(basename, raw_depth, clip_value)
                if tuned is not None:
                    fused_depth = DepthProFusion.apply(raw_depth, tuned, strategy='scale_shift')

            metrics = evaluator.evaluate(basename, fused_depth, clip_value)
            if metrics is not None:
                print(
                    '  Tractor metrics -> '
                    f"abs_rel: {metrics['abs_rel']:.4f}, "
                    f"rmse: {metrics['rmse']:.4f}, "
                    f"delta1: {metrics['delta1']:.4f}, "
                    f"suggested_max_depth: {metrics['suggested_max_depth']:.2f}m"
                )
                if clip_value is None:
                    clip_value = metrics['suggested_max_depth']

        if clip_value is not None:
            fused_depth = np.clip(fused_depth, 0.0, clip_value)

        if args.save_absolute:
            np.save(os.path.join(args.outdir, f'{basename}_abs_depth.npy'), fused_depth)

        if args.grayscale:
            normalised = (normalise_depth(fused_depth) * 255.0).astype(np.uint8)
            depth_display = np.repeat(normalised[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = colourise(fused_depth, cmap)

        output_path = os.path.join(args.outdir, f'{basename}.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth_display)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_display])
            cv2.imwrite(output_path, combined_result)

    if evaluator is not None:
        summary = evaluator.summarise()
        if summary is not None:
            print('Tractor scenario summary:')
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f'  {key}: {value:.4f}')
                else:
                    print(f'  {key}: {value}')
    if clip_value is not None:
        print(f'Final depth clipping threshold: {clip_value:.2f} m')


if __name__ == '__main__':
    main()
