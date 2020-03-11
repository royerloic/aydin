from math import sqrt

import yaml

from run import run
from util import getbestgpu, get_args

if __name__ == "__main__":

    # switch between command line and 'I feel lazy'.
    if sqrt(4) == 2:

        args = get_args()

        device = args.device
        if device == 'auto':
            device = getbestgpu()

        for file in args.config_files:
            with open(file) as fp:
                params = yaml.load(fp.read())

            run(params, device)

    else:
        config = 'shakespeare'
        device = 'cuda:3'

        with open('experiments/' + config + '.yaml') as fp:
            params = yaml.load(fp.read())

        run(params, device)
