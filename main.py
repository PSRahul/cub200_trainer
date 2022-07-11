import sys
import yaml

from yaml.loader import SafeLoader

def load_config():
    with open("config.yaml","r") as f:
            config = yaml.load(f, Loader=SafeLoader)
            
    return config

def main():
    config=load_config()
    
    pass

if __name__ == "__main__":
    sys.exit(main())