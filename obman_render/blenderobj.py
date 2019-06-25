import bpy


def load_obj_model(geometry_path, join=True):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.obj(filepath=geometry_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.scene.objects.active = obj
    bpy.ops.object.join()
    return obj


def delete_obj_model(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.ops.object.delete()
