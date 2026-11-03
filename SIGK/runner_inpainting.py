import argparse


def main():
    parser = argparse.ArgumentParser(description="Train inpainting U-Net")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=75,
        help="Number of training epochs (default: 75)",
    )
    parser.add_argument(
        "--hole-size",
        type=int,
        default=32,
        dest="hole_size",
        help="Square hole size in pixels (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        dest="batch_size",
        help="DataLoader batch size (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        dest="num_workers",
        help="DataLoader worker processes (default: 4; use 0 on Mac/MPS)",
    )
    args = parser.parse_args()
    from SIGK.engine_inpainting import main as train

    train(
        epochs=args.epochs,
        hole_size=args.hole_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
