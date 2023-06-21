########################################################################################
import gradio as gr

import cv2
import matplotlib
import matplotlib.cm
import mediapipe as mp
import numpy as np
import os
import pygltflib
import shutil
import struct
import tempfile
import torch

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image
from quads import QUADS
from typing import List, Mapping, Optional, Tuple, Union
from utils import colorize

class face_image_to_face_mesh:
    def __init__(self):
        self.zoe_me = True
        self.css = ("""
            #img-display-container {
                max-height: 50vh;
                }
            #img-display-input {
                max-height: 40vh;
                }
            #img-display-output {
                max-height: 55vh;
                max-width:  55vh;
                width:auto;
                height:auto
                }
        """)

    def demo(self):
        if self.zoe_me:
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.zoe = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

        demo = gr.Blocks(css=self.css)
        with demo:
            gr.Markdown("""
                # Face Image to Face Quad Mesh

                Uses MediaPipe to detect a face in an image and convert it to a quad mesh.
                Saves to OBJ since gltf does not support quad faces. The 3d viewer has Y pointing the opposite direction from Blender, so ya hafta spin it.

                The face depth with Zoe can be a bit much and without it is a bit generic. In blender you can fix this just by snapping to the high poly model.
            """)

            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(label="Input image", type="numpy", source="upload")

                    gr.Examples( examples=[
                        'examples/blonde-00019-1421846474.png',
                        'examples/dude-00110-1227390728.png',
                        'examples/granny-00056-1867315302.png',
                        'examples/tuffie-00039-499759385.png',
                        'examples/character.png',
                    ], inputs=[upload_image] )
                    upload_image_btn = gr.Button(value="Detect faces")
                    if self.zoe_me:
                        with gr.Group():
                            use_zoe = gr.Checkbox(label="Use ZoeDepth for Z", value=True)
                            gr.Textbox(show_label=False, value="Override the MediaPipe depth with ZoeDepth.")
                            zoe_scale = gr.Slider(label="Zoe Scale", value=1.44, minimum=0.0, maximum=3.3, step=0.1)
                            gr.Textbox(show_label=False, value="How much to scale the ZoeDepth. 2x is pretty dramatic...")
                    else:
                        use_zoe = False
                        zoe_scale = 0
                    with gr.Group():
                        min_detection_confidence = gr.Slider(label="Min detection confidence", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                        gr.Textbox(show_label=False, value="Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful.")
                    with gr.Group():
                        gr.Markdown(
                        """
                        The initial workflow I was imagining was:

                        1. sculpt high poly mesh in blender
                        2. snapshot the face
                        3. generate the mesh using the mediapipe stuff
                        4. import the low poly mediapipe face
                        5. snap the mesh to the high poly model
                        6. model the rest of the low poly model
                        7. bake the normal / etc maps to the low poly face model
                        8. it's just that easy 😛

                        Ideally it would be a plugin...
                        """)

                with gr.Column():
                    with gr.Group():
                        num_faces_detected = gr.Number(label="Number of faces detected", value=0)
                        output_image = gr.Image(label="Output image",elem_id='img-display-output')
                        output_mesh = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",elem_id='img-display-output')
                        depth_image = gr.Image(label="Depth image",elem_id='img-display-output')

            upload_image_btn.click(
                fn=self.detect, 
                inputs=[upload_image, min_detection_confidence,use_zoe,zoe_scale], 
                outputs=[output_mesh, output_image, depth_image, num_faces_detected]
            )
        demo.launch()


    def detect(self, image, min_detection_confidence, use_zoe,zoe_scale):
        width  = image.shape[1]
        height = image.shape[0]
        ratio  = width / height

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
            
        mesh = "examples/jackiechan.obj"

        if self.zoe_me and use_zoe:
            depth = self.zoe.infer_pil(image)
            print( f'type of depth is {type(depth)}' )
            idepth = colorize(depth, cmap='gray_r')
        else:
            depth = None
            idepth = image

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return mesh, image, idepth, 0

            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mesh = self.toObj(image=image, width=width, height=height, ratio=ratio, landmark_list=face_landmarks, depth=depth, zoe_scale=zoe_scale)

                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            return mesh, annotated_image, idepth, 1

    def toObj( self, image: np.ndarray, width:int, height:int, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList, depth: np.ndarray, zoe_scale: float):
        print( f'you have such pretty hair' )

        obj_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        mtl_file = tempfile.NamedTemporaryFile(suffix='.mtl', delete=False)
        png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)

        ############################################

        lines = []
        lines.append( f'mtllib {os.path.basename(mtl_file.name)}' )

        (points,coordinates) = self.landmarksToPoints( width, height, ratio, landmark_list, depth, zoe_scale )
        for point in points:
            lines.append( "v " + " ".join([str(value) for value in point]) )

        for coordinate in coordinates:
            lines.append( "vt " + " ".join([str(value) for value in coordinate]) )

        for quad in QUADS:
            normal = self.totallyNormal( points[ quad[ 0 ] -1 ], points[ quad[ 1 ] -1 ], points[ quad[ 2 ] -1 ] )
            lines.append( "vn " + " ".join([str(value) for value in normal]) )

        lines.append( 'usemtl MyMaterial' )

        quadIndex = 0
        for quad in QUADS:
            quadIndex = 1 + quadIndex
            if True:
                lines.append( "f " + " ".join([f'{vertex}/{vertex}/{quadIndex}' for vertex in quad]) )
            else:
                lines.append( "f " + " ".join([str(vertex) for vertex in quad]) )


        out = open( obj_file.name, 'w' )
        out.write( '\n'.join( lines ) + '\n' )
        out.close()
        shutil.copy(obj_file.name, "/tmp/lol.obj")

        ############################################

        lines = []
        lines.append( 'newmtl MyMaterial' )
        lines.append( f'Ka 1.000 1.000 1.000     # white' )
        lines.append( f'Kd 1.000 1.000 1.000     # white' )
        lines.append( f'Ks 0.000 0.000 0.000     # black (off)' )
        lines.append( f'map_Ka {os.path.basename(png_file.name)}' )
        lines.append( f'map_Kd {os.path.basename(png_file.name)}' )

        out = open( mtl_file.name, 'w' )
        out.write( '\n'.join( lines ) + '\n' )
        out.close()
        shutil.copy(mtl_file.name, "/tmp/lol.mtl")

        ############################################

        cv2.imwrite(png_file.name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        shutil.copy(png_file.name, "/tmp/lol.png")

        ############################################

        print( f'I know it is special to you so I saved it to {obj_file.name} since we are friends' )
        return obj_file.name

    def landmarksToPoints( self, width: int, height: int, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList, depth: np.ndarray, zoe_scale: float ):
        points      = [] # 3d vertices
        coordinates = [] # 2d texture coordinates
        mins = [+np.inf] * 3
        maxs = [-np.inf] * 3
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                    idk_what_to_do_for_this = True
            x, y = _normalized_to_pixel_coordinates(landmark.x,landmark.y,width,height)
            coordinates.append( [x/width,1-y/height] )
            if depth is not None:
                landmark.z = depth[y, x] * zoe_scale
            point = [landmark.x * ratio, -landmark.y, -landmark.z];
            for pidx,value in enumerate( point ):
                mins[pidx] = min(mins[pidx],value)
                maxs[pidx] = max(maxs[pidx],value)
            points.append( point )

        mids = [(min_val + max_val) / 2 for min_val, max_val in zip(mins, maxs)]
        for idx,point in enumerate( points ):
            points[idx] = [(val-mid) for val, mid in zip(point,mids)]

        print( f'mins: {mins}' )
        print( f'mids: {mids}' )
        print( f'maxs: {maxs}' )
        return (points,coordinates)

    def totallyNormal(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        return normal.tolist()    


face_image_to_face_mesh().demo()

# EOF
########################################################################################
