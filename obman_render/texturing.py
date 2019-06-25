import os
import pickle
import shutil

import numpy as np


# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob):
    import bpy
    print('creating segmentation')
    materials = {}
    vgroups = {}
    with open('assets/segms/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = pickle.load(f)
    bpy.context.scene.objects.active = ob
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    sorted_parts = [
        'hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
        'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase',
        'rightToeBase', 'neck', 'leftShoulder', 'rightShoulder', 'head',
        'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand',
        'rightHand', 'leftHandIndex1', 'rightHandIndex1'
    ]
    part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)

        # Duplicates sh_material to all body parts
        mater = bpy.data.materials['Material'].copy()
        materials[part] = mater

        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return (materials)


def create_sh_material(tree, sh_path, down_scale=1):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # rgb = tree.nodes.new('ShaderNodeRGB')
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeMapping')
    uv_xform.location = -600, 400
    uv_xform.scale = (down_scale, down_scale, down_scale)

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = sh_path  #'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_xform.inputs[0])
    tree.links.new(uv_xform.outputs[0], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    # tree.links.new(rgb.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])


def add_obj_texture(mater,
                    obj_texture_path,
                    sh_path,
                    down_scale=1,
                    tmp_suffix='',
                    generated_uv=True):
    import bpy
    tmp_folder = 'tmp_{}/osl'.format(tmp_suffix)
    os.makedirs(tmp_folder, exist_ok=True)
    filename = str(np.random.randint(0, 1000000)) + '.osl'
    while os.path.exists(os.path.join(tmp_folder, filename)):
        filename = str(np.random.randint(0, 1000000)) + '.osl'
    obj_sh_path = os.path.join(tmp_folder, filename)
    shutil.copy(sh_path, obj_sh_path)

    # Initialize sh texture
    mater.use_nodes = True
    mater.pass_index = 100
    tree = mater.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)

    # rgb = tree.nodes.new('ShaderNodeRGB')
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeMapping')
    uv_xform.location = -600, 400
    uv_xform.scale = (down_scale, down_scale, down_scale)

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    # uv_im.projection = 'TUBE'
    uv_im.location = -400, 400

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = obj_sh_path  #'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    if generated_uv:
        tree.links.new(uv.outputs[0], uv_xform.inputs[0])
    else:
        tree.links.new(uv.outputs[2], uv_xform.inputs[0])
    tree.links.new(uv_xform.outputs[0], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    # tree.links.new(rgb.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

    tex_img = bpy.data.images.load(obj_texture_path)
    tree.nodes['Image Texture'].image = tex_img
    return obj_sh_path


def get_sh_coeffs(ambiant_mean=0.7, ambiant_max_add=0.9):
    # randomize light. Ambient light needs a minimum
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = ambiant_mean + ambiant_max_add * np.random.rand(
    )  # was .3 and .7. first coeff is ambient (.3 is the minimum, rest is uniform distributed, higher means brighter)
    sh_coeffs[1] = -.7 * np.random.rand()
    return sh_coeffs


def set_sh_coeffs(scs, sh_coeffs):
    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish + 1].default_value = coeff


def initialize_texture(obj, texture_zoom=1, tmp_suffix=''):
    import bpy
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    scene.cycles.film_transparent = True
    mater = bpy.data.materials['Material']
    mater.use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True
    scene.cycles.film_transparent = True
    node_tree = mater.node_tree

    # The spherical harmonics material needs a script to be loaded and compiled!
    sh_dst = 'assets/lights/sh.osl'

    # Make temporary copy of osl file
    tmp_folder = 'tmp_{}/osl'.format(tmp_suffix)
    os.makedirs(tmp_folder, exist_ok=True)
    filename = str(np.random.randint(0, 1000000)) + '.osl'
    while os.path.exists(os.path.join(tmp_folder, filename)):
        filename = str(np.random.randint(0, 1000000)) + '.osl'
    sh_path = os.path.join(tmp_folder, filename)
    shutil.copy(sh_dst, sh_path)
    create_sh_material(node_tree, sh_path, down_scale=texture_zoom)

    materials = create_segmentation(obj)

    scs = []
    for mname, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].update()
    return scs, materials, sh_path
