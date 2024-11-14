import os

APPLICATIONS = 'Applications'
HOME = '.flamingo'


def get_app_path(app_name: str) -> str:
    home_dir = os.path.expanduser("~")
    applications_dir = os.path.join(home_dir, HOME, APPLICATIONS)
    app_path = os.path.join(applications_dir, app_name)

    # Create directories if they don't exist
    os.makedirs(app_path, exist_ok=True)

    return app_path

def get_subfolders(folder_path: str) -> list[str]:
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    subfolders = sorted(subfolders)
    return subfolders