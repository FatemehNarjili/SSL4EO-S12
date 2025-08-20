import json
import os

TRAINING_STATE_FILE = "training_state.json"

class TrainingStateManager:
    def __init__(self):
        self.file_path = TRAINING_STATE_FILE
        self.state = {}
        self._load_state()

    def _load_state(self):
        """Load the training state from the JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.state = json.load(file)
        else:
            self.state = {"last_logged_epoch": 0, "best_f1": 0.0}
            self._save_state()

    def _save_state(self):
        """Save the current state to the JSON file."""
        with open(self.file_path, "w") as file:
            json.dump(self.state, file)

    def get_state_value(self, key, default=None):
        """Retrieve a value from the training state."""
        return self.state.get(key, default)

    def set_state_value(self, key, value):
        """Set a value in the training state."""
        self.state[key] = value
        self._save_state()
