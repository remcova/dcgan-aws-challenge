import argparse


class DCGANArgParser(argparse.ArgumentParser):
    def __init__(self):
        description = "Run this training script (dcgan.py) on multiple Gaudi cores, by using the following mpirun command: \
            mpirun -np 8 python3 dcgan.py"
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
        )
        self.add_argument("--batch_size", "-b", type=int, default=12, help="Batch size")
        self.add_argument(
            "--epochs", "-e", type=int, default=30000, help="Amount of Epochs"
        )
        self.add_argument(
            "--show_samples_interval",
            type=int,
            default=10,
            help="Show generated samples during training every X epoch",
        )
        self.add_argument(
            "--save_num_samples",
            type=int,
            default=10,
            help="Save X amount of generated samples",
        )
        self.add_argument(
            "--use_hpu",
            dest="use_hpu",
            action="store_true",
            help="If set, HPU will be used",
        )
        self.add_argument(
            "--use_horovod",
            help="If set, Horovod will be used for distributed training",
            action="store_true",
        )
        self.add_argument(
            "--restore",
            dest="restore",
            action="store_true",
            help="If set, model will be restored from checkpoint",
        )
        self.add_argument(
            "--log_device_placement",
            dest="log_device_placement",
            action="store_true",
            help="If set, log Tensorflow operations",
        )
