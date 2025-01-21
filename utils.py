import json
import pickle

def load_file(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".pkl"):
        return pickle.load(open(file_path, "rb"))