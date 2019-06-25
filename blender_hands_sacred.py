import json
import sys

root = '.'
sys.path.insert(0, root)

from blender_hands import ex as hand_ex

recover_json_string = ' '.join(sys.argv[sys.argv.index('--') + 1:])
json_config = json.loads(recover_json_string)

# Generate hands only by sampling from random hand poses
r = hand_ex.run(config_updates=json_config)
