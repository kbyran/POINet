import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launcher")
    parser.add_argument("--task", default="train", type=str,
                        choices=["train", "val", "test"], help="task to run")
    parser.add_argument("--config", required=True, help="config to load")
    parser.add_argument("--epoch", type=int, help="epoch to val or test")
    args = parser.parse_args()

    if args.task == "train":
        from poi.apis.train import train_net
        train_net(config=args.config)
    elif args.task == "val":
        from poi.apis.test import test_net
        test_net(config=args.config, stage="val", epoch=args.epoch)
    elif args.task == "test":
        from poi.apis.test import test_net
        test_net(config=args.config, stage="test", epoch=args.epoch)
