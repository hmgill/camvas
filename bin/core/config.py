
import pathlib
from confection import registry, Config


class ConfigReader():

    def __init__(self):
        pass


    def read_config(self, config_path):
        config = Config().from_disk(config_path)
        output_config = {}

        for key, value in config.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                output_config[key] = pathlib.Path(value)
            elif isinstance(value, dict):
                output_config[key] = self.process_nested_dict(value)
            elif isinstance(value, list):
                output_config[key] = self.process_nested_list(value)
            else:
                output_config[key] = value
        return output_config


    def process_nested_dict(self, nested_dict: dict) -> dict:
        result = {}
        for key, value in nested_dict.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                result[key] = pathlib.Path(value)
            elif isinstance(value, dict):
                result[key] = self.process_nested_dict(value)
            elif isinstance(value, list):
                result[key] = self.process_nested_list(value)
            else:
                result[key] = value
        return result

    def process_nested_list(self, nested_list: list) -> list:
        result = []
        for item in nested_list:
            if isinstance(item, str) and ("/" in item or "\\" in item):
                result.append(pathlib.Path(item))
            elif isinstance(item, dict):
                result.append(self.process_nested_dict(item))
            elif isinstance(item, list):
                result.append(self.process_nested_list(item))
            else:
                result.append(item)
        return result
