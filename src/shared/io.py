import os
import time


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time


def get_project_root():
    return os.path.abspath('../../../')


def get_log_root():
    return os.path.join(get_project_root(), 'log')


def get_data_root():
    return os.path.join(get_project_root(), 'datasets')


def get_checkpoint_root():
    return os.path.join(get_project_root(), 'checkpoints')


def get_output_root():
    return os.path.join(get_project_root(), 'out')


def main():
    print(get_log_root())


if __name__ == '__main__':
    main()
