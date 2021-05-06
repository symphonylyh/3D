# entry for run command `blender.exe --background --python blender_LOD_merge.py`

import subprocess

blender_path = r"D:\Software Installation\Blender\blender.exe" # location of blender.exe

subprocess.run([blender_path, '--background', '--python', 'blender_LOD_merge.py'])