import os

def print_tree(start_path, indent=""):
    for item in sorted(os.listdir(start_path)):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(f"{indent}├── {item}/")
            print_tree(path, indent + "│   ")
        else:
            print(f"{indent}├── {item}")

print_tree(".")
