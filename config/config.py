import json

class Configuration:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self._load_config()
        
    def _load_config(self):
        self.config_json = json.load(self.config_file_path)

HOME_DIR = "../HOME_DIR"