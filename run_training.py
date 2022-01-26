import sys
from lib.training.execute import get_configs_from_args, execute

if __name__ == '__main__':
    config = get_configs_from_args(sys.argv)
    execute('train', config)
