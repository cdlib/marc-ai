import yaml


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)
