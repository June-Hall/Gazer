#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))

from tools.infer_video_720p_vbench import (
    InferencePipe,
    create_args,
    logger,
    perform_inference,
)
from tools.run_infinity import save_video


def parse_args():
    parser = argparse.ArgumentParser(description="InfinityStar 720p single demo generation")

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to the InfinityStar checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/InfinityStar/demo_self_reflection/demo.mp4",
        help="Path for the generated demo video",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a person drinking coffee in a cafe",
        help="Text prompt for the demo video",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--video_frames", type=int, default=81, help="Number of generated frames")
    parser.add_argument("--cfg", type=float, default=34, help="CFG/APG guidance strength")
    parser.add_argument("--tau_image", type=float, default=1.0, help="Image scale temperature")
    parser.add_argument("--tau_video", type=float, default=0.4, help="Video scale temperature")
    parser.add_argument("--image_path", type=str, default=None, help="Optional reference image for I2V")

    parser.add_argument(
        "--disable_self_reflection",
        action="store_true",
        help="Disable VLM self-reflection during demo generation",
    )
    parser.add_argument(
        "--reflection_start_scale",
        type=float,
        default=0.1,
        help="Scale progress at which self-reflection starts",
    )
    parser.add_argument(
        "--reflection_end_scale",
        type=float,
        default=0.9,
        help="Scale progress at which self-reflection ends",
    )
    parser.add_argument(
        "--reflection_interval",
        type=int,
        default=2,
        help="Self-reflection interval in scale steps",
    )
    parser.add_argument(
        "--use_only_enhanced_prompt",
        action="store_true",
        default=False,
        help="Use only the latest enhanced prompt after self-reflection",
    )

    args = parser.parse_args()
    args.enable_self_reflection = not args.disable_self_reflection
    return args


def main():
    cli_args = parse_args()

    logger.info("=" * 60)
    logger.info("InfinityStar 720p demo generation")
    logger.info("=" * 60)
    for key, value in vars(cli_args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    if not os.path.exists(cli_args.checkpoints_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {cli_args.checkpoints_dir}")

    args = create_args(cli_args.checkpoints_dir, cli_args)
    pipe = InferencePipe(args)

    result = perform_inference(
        pipe=pipe,
        prompt=cli_args.prompt,
        seed=cli_args.seed,
        args=args,
        image_path=cli_args.image_path,
    )

    save_video(result["output"], fps=cli_args.fps, save_filepath=cli_args.output_path)
    logger.info(f"Demo generation done: {osp.abspath(cli_args.output_path)}")
    logger.info(f"Elapsed time: {result['elapsed_time']:.2f}s")


if __name__ == "__main__":
    main()
