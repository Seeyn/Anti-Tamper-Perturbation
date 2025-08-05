import argparse


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--input_dir', type=str, required=True,default=None,
                                help='The directory of input images')
    parent_parser.add_argument('--output_dir', type=str,required=True, default=None,
                                help='The directory of output authorized images')
    parent_parser.add_argument('--device', required=True, type=str,
                                help='device for training, e.g., cuda:0')
    parent_parser.add_argument('--gamma', default=0.5, type=float,
                                help='deciding the 0-1 mask percentage')
    parent_parser.add_argument("--input-mask",
                        default=None,type=str,
                        help="input-mask path")
    parent_parser.add_argument("--noise-type",
                        default=None,type=str,
                        help="implement noise type")
    parent_parser.add_argument("--noise-args",
                        default=0.5, type=float,
                        help="noise args")
    parent_parser.add_argument("--pixel-space",
                        action="store_true",
                        help="pixel space or not")
    args = parent_parser.parse_args()
    return args
    