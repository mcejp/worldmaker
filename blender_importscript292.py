import bpy
import sys
 
argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

FILENAME = argv[0]

bpy.ops.object.delete()
bpy.ops.import_mesh.ply(filepath=FILENAME)

obj, = bpy.context.selected_objects

mat = bpy.data.materials.new("VertCol")
mat.use_nodes = True
node_tree = mat.node_tree
nodes = node_tree.nodes

bsdf = nodes.get("Principled BSDF") 
assert bsdf

vcol = nodes.new(type="ShaderNodeVertexColor")
vcol.layer_name = "Col"

node_tree.links.new(vcol.outputs[0], bsdf.inputs[0])
bsdf.inputs['Specular'].default_value = 0

obj.data.materials.append(mat)

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100
bpy.ops.view3d.camera_to_view_selected()
