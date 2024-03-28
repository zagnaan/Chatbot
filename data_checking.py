import torch

# Load the .pth file
FILE = "data.pth"
data = torch.load(FILE)

# Print the keys to see what's included in the file
print("Keys in the .pth file:", data.keys())

# Example: Print the model state dictionary
print("Model State Dictionary:", data["model_state"])

# Example: Print other information like input size, output size, etc.
print("Input Size:", data["input_size"])
print("Output Size:", data["output_size"])
print("Hidden Size:", data["hidden_size"])
print("All Words:", data["all_words"])
print("Tags:", data["tags"])
