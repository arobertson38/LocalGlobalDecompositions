import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='Results/logs/40', \
        help='The location containing the config file to clean.')

args = parser.parse_args()

# checking and cleaning the file
if os.listdir(args.folder).__contains__('config.yml'):
    with open(os.path.join(args.folder, 'config.yml'), 'r') as f:
        lines = f.readlines()

    lines = [line.replace('!', '#') for line in lines]

    with open(os.path.join(args.folder, 'config.yml'), 'w') as f:
            f.writelines(lines)
        
