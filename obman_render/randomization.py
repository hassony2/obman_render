import bpy
import math
from mathutils import Vector
import random

def move_on_sphere(obj, center_location, target_obj=None, radius=2):
    """Moves the obj to a random point on the sphere of center center
    with given radius
    """
    if target_obj is not None:
        tracking = obj.constraints.new('TRACK_TO')
        tracking.target = target_obj
        tracking.track_axis ='TRACK_NEGATIVE_Z'
        tracking.up_axis = 'UP_Y'
    else:
        tracking = None

    phi_deg = random.randint(0, 180)
    phi = math.radians(phi_deg)
    theta_deg = random.randint(0, 360)
    theta = math.radians(theta_deg)

    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    old_x, old_y, old_z = center_location
    new_x, new_y, new_z = old_x + x, old_y + y, old_z + z
    obj.location = Vector((new_x, new_y, new_z))
    return tracking
