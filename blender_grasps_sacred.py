import json
import sys

root = '.'
sys.path.insert(0, root)

from blender_grasps import ex as grasp_ex

recover_json_string = ' '.join(sys.argv[sys.argv.index('--') + 1:])
json_config = json.loads(recover_json_string)

# Generate grasps
r = grasp_ex.run(config_updates=json_config)
