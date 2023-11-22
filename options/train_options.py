

import argparse
import datetime
import os
import time


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="self-supervised view synthesis"
        )
        self.add_data_parameters()
        self.add_train_parameters()
        self.add_model_parameters()

    def add_model_parameters(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument(
            "--model_type",
            type=str,
            default="zbuffer_pts",
            choices=(
                "zbuffer_pts",
                "z_buffermodel_w_hfrefine",
                "z_buffermodel_w_depth_estimation",
                "z_buffermodel_w_depth_estimation_w_hfrefine",
            ),
            help='Model to be used.'
        )

        model_params.add_argument(
            "--accumulation",
            type=str,
            default="wsum",
            choices=("wsum", "wsumnorm", "alphacomposite"),
            help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)"
        )


        model_params.add_argument(
            "--splatter",
            type=str,
            default="xyblending",
            choices=("xyblending"),
        )
        model_params.add_argument("--rad_pow", type=int, default=2,
            help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
        # model_params.add_argument("--num_views", type=int, default=2,



        model_params.add_argument(
            "--learn_default_feature", action="store_true", default=True
        )


        model_params.add_argument("--pp_pixel", type=int, default=128,
            help='K: the number of points to conisder in the z-buffer.'
        )
        model_params.add_argument("--tau", type=float, default=1.0,
            help='gamma: the power to raise the distance to.'
        )
       
        model_params.add_argument(
            "--radius",
            type=float,
            default=4,
            help="Radius of points to project",
        )
        model_params.add_argument(
            "--depth_radius",
            type=float,
            default=4,
            help="Radius of points to project",
        )




        
    def add_data_parameters(self):
        dataset_params = self.parser.add_argument_group("data")
        dataset_params.add_argument("--dataset", type=str, default="tanksandtemples")
        dataset_params.add_argument("--min_z", type=float, default=0.5)
        dataset_params.add_argument("--max_z", type=float, default=10.0)
        dataset_params.add_argument("--W", type=int, default=256)
       
        dataset_params.add_argument(
            "--use_rgb_features", action="store_true", default=False
        )

        dataset_params.add_argument(
            "--normalize_image", action="store_true", default=False
        )

    def add_train_parameters(self):

        training = self.parser.add_argument_group("training")
        training.add_argument("--num_workers", type=int, default=0)
        training.add_argument("--start-epoch", type=int, default=0)
        training.add_argument("--lr", type=float, default=1e-3)
        training.add_argument("--beta1", type=float, default=0)
        training.add_argument("--beta2", type=float, default=0.9)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--init", type=str, default="")

      

        training.add_argument(
            "--losses", type=str, nargs="+", default=['1.0_content','10.0_l1']
        )

        training.add_argument(
            "--load-old-model", action="store_true", default=False
        )

        training.add_argument("--old_model", type=str, default="")
      
        training.add_argument("--resume", action="store_true", default=False)

        training.add_argument(
            "--log-dir",
            type=str,
            default="./checkpoint/%s/",
        )

        training.add_argument("--batch-size", type=int, default=16)
        training.add_argument("--continue_epoch", type=int, default=0)
        training.add_argument("--max_epoch", type=int, default=500)
        training.add_argument("--folder_to_save", type=str, default="outpaint")
        training.add_argument(
            "--model-epoch-path",
            type=str,
            default="/%s/%s/lr%s_bs%s/spl%s/",
        )
        training.add_argument(
            "--run-dir",
            type=str,
            default="/%s/%s/lr%s_bs%s/spl%s/",
        )
        training.add_argument("--gpu_ids", type=str, default="7")
        training.add_argument('--path', help="dataset for evaluation")


        training.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
        training.add_argument('--n_ratio', type=float, default=1.1, 
                        help='ratio to N fot the num of anchor points')


    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)


def get_timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return st


def get_log_path(timestamp, opts):
    return (
        opts.log_dir % (opts.dataset)
        + '/%s'+opts.run_dir
        % (
            timestamp,
            opts.folder_to_save,
            opts.lr,
            opts.batch_size,
            opts.splatter,
        )
    )


def get_model_path(timestamp, opts):
    model_path = opts.log_dir % (opts.dataset) + opts.model_epoch_path % (
        timestamp,
        opts.folder_to_save,
        opts.lr,
        opts.batch_size,
        opts.splatter,
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path + "/model_epoch.pth"
