
#%% 1. Is the ODM engine reachable?
from pyodm import Node
from config import *                     # gives BASE, OUT

node = Node("localhost", 3000)           # connect to ODM running in Docker
print(node.info())                       # prints the ODM version if it's running

#%% 2. Find the photos
photos = [str(p) for p in sorted((BASE / "images_nadir_RGB").rglob("*.JPG"))]
print(len(photos), "photos")             # should be 86

#%% 3. Start the job and show progress
task = node.create_task(photos, {
    "orthophoto-resolution": 1.6,        # cm per pixel, same as their orthomosaic
    "dsm": True,                         # surface model (with trees)
    "dtm": True,                         # bare-ground model (for tree heights)
    "feature-quality": "medium",         # match points on smaller copies of photos (less RAM)
    "pc-quality": "medium",              # lighter 3D point cloud (less RAM)
})
print("uploaded, processing...")
task.wait_for_completion(status_callback=lambda info: print(info.status, f"{info.progress:.0f}%"))
task.download_assets(str(OUT / "my_odm"))   # results go into outputs/my_odm
print("done")











































