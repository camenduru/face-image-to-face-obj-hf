import gradio as gr

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

import numpy as np
from mediapipe.framework.formats import landmark_pb2
from typing import List, Mapping, Optional, Tuple, Union

import pygltflib
import struct
import tempfile

# ok... I goofed one of them :-(
QUADS = [
	[300, 334, 333, 298] , [  1,  12, 303, 268] , [234, 233, 122, 129] , [270, 304, 305, 271] , [246, 129, 115, 189] ,
	[112, 118, 229,  32] , [104,  55,  69, 105] , [228,  35, 128, 235] , [120, 102, 101, 121] , [ 74,  73,  38,  40] ,
	[ 71,  47,  54,  64] , [135, 132, 116, 221] , [335, 294, 299, 334] , [ 73,  12,   1,  38] , [ 42,  43,  81,  82] ,
	[166,  93,  41,  40] , [122, 233, 232, 121] , [215, 213, 217, 208] , [183,  84,  85, 182] , [376, 308, 321, 322] ,
	[ 30, 161, 160,  28] , [ 57,  29, 159, 158] , [ 84, 202, 201,  19] , [117, 144,  35, 228] , [204, 207,  93, 166] ,
	[139, 216,  59, 173] , [276, 282,   6,   5] , [ 25, 145, 164, 111] , [292, 307, 308, 376] , [143, 127,  48, 101] ,
	[419, 422, 429, 263] , [147,  44, 107,  92] , [ 17,  86,  85,  18] , [ 78,  77,  62, 147] , [127, 210, 199, 218] ,
	[397, 378, 401, 370] , [166,  40,  38, 168] , [245, 234, 129, 246] , [ 31, 248, 247, 162] , [ 34, 247, 248, 131] ,
	[175, 218, 199, 237] , [418, 352, 413, 466] , [125, 114, 226,  47] , [225, 224,  53,  54] , [ 99,  65, 103, 130] ,
	[193, 215, 208, 188] , [219,  80, 240, 238] , [134, 156, 113, 244] , [345, 361, 364, 441] , [141, 171, 150, 177] ,
	[400, 413, 352, 420] , [119, 230, 229, 118] , [282, 276, 441, 364] , [ 71,  64,  69,  72] , [315, 314, 407, 406] ,
	[222, 190, 194,  56] , [114, 248,  31, 226] , [106,  53,  66,  67] , [236,  60, 167, 220] , [108,  56,   9,  10] ,
	[ 67,  66,  56, 108] , [ 69,  64, 106, 105] , [120, 119,  51, 102] , [242, 126,  45, 238] , [  6, 196,   4,  52] ,
	[143, 130, 210, 127] , [ 34, 131,  26,   8] , [323, 271, 410, 411] , [ 33, 195, 205, 212] , [ 37, 102,  51, 206] ,
	[195, 202,  84, 183] , [238, 240, 239, 242] , [ 26, 111, 164,   8] , [225,  54,  47, 226] , [154, 146,  24,  23] ,
	[211, 203, 213, 215] , [246, 194, 190, 245] , [425, 336, 407, 419] , [318, 317, 404, 403] , [ 33, 212, 171, 141] ,
	[ 12,  73,  39,  13] , [208, 217, 207, 206] , [238, 221, 116, 219] , [ 46, 221, 238,  45] , [184,  43,  75, 185] ,
	[209, 202, 195,  33] , [269, 272, 304, 303] , [214, 148, 178, 216] , [235,  94, 138, 228] , [ 67, 108, 109,  70] ,
	[  7, 352, 418, 169] , [193, 188, 148, 214] , [ 97,  63,  77,  78] , [125,  47,  71, 157] , [317,  16,  17, 316] ,
	[115, 129, 122,  48] , [148, 124, 138, 178] , [252, 285, 333, 334, 299, 302] , [181,  86,  87, 180] , [290, 393, 291, 306] ,
	[180,  87,  88, 179] , [106,  64,  54,  53] , [119, 118, 124,  51] , [146, 145,  25,  24] , [325, 319, 320, 326] ,
	[123, 189, 175, 197] , [293, 309, 325, 326] , [150, 171, 170, 151] , [178, 138,  94, 133] , [328, 295, 456, 461] ,
	[361, 421, 457, 364] , [336, 274, 376, 322] , [396, 395, 431, 432] , [ 13,  39,  83,  14] , [278, 330, 350, 351] ,
	[191,  57, 158, 174] , [117, 112,  36, 144] , [224, 223,  66,  53] , [140,  72,  22, 163] , [163, 128,  35, 140] ,
	[366, 365, 395, 380] , [219, 116,  49, 220] , [430, 359, 372, 356] , [157, 144,  36, 125] , [377, 353, 281, 412] ,
	[125,  36, 227, 114] , [355,  20,  95, 371] , [120, 231, 230, 119] , [249, 457, 400, 420] , [162, 161,  30,  31] ,
	[ 46,  45,   2,   5] , [141, 172, 209,  33] , [394, 392, 328, 327] , [ 32,  26, 131, 227] , [300, 298, 339, 338] ,
	[395, 396, 379, 380] , [102,  37, 143, 101] , [217, 213,  58, 187] , [327,   3, 165, 394] , [242, 239,  21, 243] ,
	[186,  41,  93, 187] , [269, 303,  12,  13] , [192,  81,  43, 184] , [140,  35, 144, 157] , [223, 222,  56,  66] ,
	[189, 115, 218, 175] , [323, 427, 424, 392] , [ 37, 204, 130, 143] , [280, 430, 421, 361] , [  2, 275, 276,   5] ,
	[134, 244, 191, 174] , [241,  76,  60, 236] , [108,  10, 152, 109] , [ 27, 155, 154,  23] , [211, 215, 136, 170] ,
	[355, 275,   2,  20] , [ 90,  89,  96,  97] , [321, 320, 404, 405] , [316, 315, 406, 405] , [107,  44, 203, 205] ,
	[201, 422, 314,  19] , [153, 176, 172, 149] , [376, 274, 288, 292] , [292, 288, 411, 410] , [130, 204, 166,  99] ,
	[115,  48, 127, 218] , [327, 328, 461, 329] , [105, 106,  67,  70] , [236,  65,  99, 241] , [200, 201, 202, 209] ,
	[332, 295, 328, 359] , [100,  61,  76, 241] , [243, 142, 126, 242] , [329, 463, 371, 327] , [220, 167,  80, 219] ,
	[233,  27,  23, 232] , [190, 222,  57, 191] , [223,  29,  57, 222] , [244, 113, 234, 245] , [ 32, 229, 111,  26] ,
	[226,  31,  30, 225] , [232,  23,  24, 231] , [225,  30,  28, 224] , [114, 227, 131, 248] , [ 32, 227,  36, 112] ,
	[234, 113,  27, 233] , [230,  25, 111, 229] , [224,  28,  29, 223] , [ 95,  20, 126, 142] , [239, 240,  80,  21] ,
	[243,  21,  61, 100] , [157,  71,  72, 140] , [ 76,  61, 167,  60] , [189, 123, 194, 246] , [231,  24,  25, 230] ,
	[232, 231, 120, 121] , [121, 101,  48, 122] , [208, 206,  51, 188] , [332, 280, 279, 295] , [196, 249, 420, 198] ,
	[199, 210,  50, 132] , [177, 149, 172, 141] , [117, 124, 118, 112] , [ 28, 160, 159,  29] , [245, 190, 191, 244] ,
	[379, 396, 370, 401] , [268, 303, 304, 270] , [351, 453, 454, 358] , [ 75,  74,  40,  41] , [169, 418, 286,   9] ,
	[283, 444, 445, 284] , [397, 176, 153, 378] , [110,  68,  70, 109] , [301, 277, 354, 384] , [186,  62,  77, 185] ,
	[299, 294, 301, 302] , [ 50,  49, 116, 132] , [422, 201, 200, 429] , [304, 272, 273, 305] , [271, 323, 392, 270] ,
	[296, 443, 444, 283] , [427, 437, 428, 426] , [336, 322, 406, 407] , [ 19, 314, 315,  18] , [387, 388, 260, 258] ,
	[255, 374, 375, 254] , [314, 422, 419, 407] , [297, 335, 334, 300] , [313, 312, 272, 269] , [ 55,  22,  72,  69] ,
	[221,  46,  52, 135] , [391, 374, 255, 340] , [315, 316,  17,  18] , [372, 267, 331, 330] , [423, 274, 336, 425] ,
	[ 58,  44, 147,  62] , [ 91,  78, 147,  92] , [182,  85,  86, 181] , [423, 425, 432, 431] , [357, 265, 448, 455] ,
	[268, 270, 392, 394] , [358, 454, 465, 466] , [264, 360, 468, 467] , [264, 250, 256, 360] , [421, 430, 356, 438] ,
	[194, 123,   7, 169] , [449, 450, 348, 347] , [277, 284, 445, 446] , [241,  99,  98, 100] , [281, 331, 267, 426] ,
	[307, 292, 410, 409] , [260, 388, 389, 261] , [364, 457, 249, 282] , [338, 339,  11, 152] , [438, 344, 413, 400] ,
	[349, 451, 452, 350] , [345, 279, 280, 361] , [402, 377, 434, 436] , [367, 324, 455, 448] , [182,  92, 107, 183] ,
	[418, 414, 442, 286] , [360, 256, 262, 447] , [284, 277, 301, 294] , [291, 251, 463, 329] , [344, 358, 466, 413] ,
	[179,  89,  90, 180] , [266, 341, 346, 373] , [429, 397, 370, 263] , [296, 283, 335, 297] , [275, 355, 462, 458] ,
	[  4, 237, 135,  52] , [359, 424, 267, 372] , [386, 387, 258, 259] , [394, 165,   1, 268] , [207, 217, 187,  93] ,
	[278, 356, 372, 330] , [ 44,  58, 213, 203] , [459, 460, 458, 462] , [381, 382, 257, 253] , [266, 447, 262, 341] ,
	[399, 385, 287, 415] , [437, 433, 435, 428] , [447, 266, 354, 343] , [183, 107, 205, 195] , [ 43,  42,  74,  75] ,
	[302, 301, 384, 369] , [425, 419, 263, 432] , [295, 279, 440, 456] , [ 49,  50, 103,  65] , [ 74,  42,  39,  73] ,
	[433, 423, 431, 435] , [311, 273, 272, 312] , [353, 367, 448, 346] , [252, 302, 369, 390] , [209, 172, 176, 200] ,
	[ 56, 194, 169,   9] , [377, 412, 417, 434] , [ 90,  97,  78,  91] , [330, 331, 349, 350] , [180,  90,  91, 181] ,
	[281, 348, 349, 331] , [265, 373, 346, 448] , [324, 367, 402, 362] , [308, 326, 320, 321] , [ 16,  15,  88,  87] ,
	[266, 373, 384, 354] , [353, 347, 348, 281] , [363, 399, 415, 464] , [318,  15,  16, 317] , [356, 278, 344, 438] ,
	[ 96,  79,  63,  97] , [ 11, 110, 109, 152] , [398, 368, 365, 366] , [  2,  45, 126,  20] , [313, 269,  13,  14] ,
	[237, 199, 132, 135] , [187,  58,  62, 186] , [152,  10, 337, 338] , [ 42,  82,  83,  39] , [414, 418, 466, 465] ,
	[467, 468, 261, 389] , [  9, 286, 337,  10] , [446, 343, 354, 277] , [265, 357, 390, 369] , [436, 434, 417, 368] ,
	[170, 136, 137, 151] , [458, 441, 276, 275] , [212, 205, 203, 211] , [347, 353, 346, 341] , [284, 294, 335, 283] ,
	[452, 453, 351, 350] , [ 95,   3, 327, 371] , [450, 451, 349, 348] , [197,   4, 196, 198] , [254, 375, 381, 253] ,
	[345, 441, 458, 439] , [367, 353, 377, 402] , [449, 347, 341, 262] , [360, 447, 343, 468] , [136, 139, 173, 137] ,
	[289, 436, 368, 398] , [281, 426, 428, 412] , [288, 433, 437, 411] , [ 99, 166, 168,  98] , [142, 243, 100,  98] ,
	[175, 237,   4, 197] , [185,  75,  41, 186] , [307, 293, 326, 308] , [396, 432, 263, 370] , [286, 442, 443, 296] ,
	[428, 435, 417, 412] , [411, 437, 427, 323] , [421, 438, 400, 457] , [165,   3,  98, 168] , [279, 345, 439, 440] ,
	[391, 340, 256, 250] , [306, 291, 329, 461] , [373, 265, 369, 384] , [386, 259, 287, 385] , [435, 365, 368, 417] ,
	[251, 459, 462, 463] , [320, 319, 403, 404] , [ 17,  16,  87,  86] , [322, 321, 405, 406] , [ 85,  84,  19,  18] ,
	[433, 288, 274, 423] , [362, 402, 436, 289] , [185,  77,  63, 184] , [293, 307, 409, 408] , [392, 424, 359, 328] ,
	[352,   7, 198, 420] , [228, 138, 124, 117] , [393, 290, 456, 440] , [176, 397, 429, 200] , [220,  49,  65, 236] ,
	[424, 427, 426, 267] , [332, 359, 430, 280] , [365, 435, 431, 395] , [310, 251, 291, 393] , [355, 371, 463, 462] ,
	[ 98,   3,  95, 142] , [255, 254, 451, 450] , [415, 414, 465, 464] , [254, 253, 452, 451] , [261, 468, 343, 446] ,
	[260, 261, 446, 445] , [258, 260, 445, 444] , [454, 342, 464, 465] , [198,   7, 123, 197] , [259, 258, 444, 443] ,
	[287, 442, 414, 415] , [340, 449, 262, 256] , [340, 255, 450, 449] , [257, 342, 454, 453] , [ 61,  21,  80, 167] ,
	[310, 393, 440, 439] , [338, 337, 297, 300] , [310, 460, 459, 251] , [ 51, 124, 148, 188] , [253, 257, 453, 452] ,
	[215, 193, 139, 136] , [351, 358, 344, 278] , [113, 156, 155,  27] , [  6,  52,  46,   5] , [206, 207, 204,  37] ,
	[249, 196,   6, 282] , [216, 178, 133,  59] , [286, 296, 297, 337] , [382, 383, 342, 257] , [287, 259, 443, 442] ,
	[211, 170, 171, 212] , [306, 461, 456, 290] , [104, 105,  70,  68] , [271, 305, 409, 410] , [460, 310, 439, 458] ,
	[214, 216, 139, 193] , [317, 316, 405, 404] , [181,  91,  92, 182] , [  1, 165, 168,  38] , [363, 464, 342, 383] ,
	[210, 130, 103,  50] , [305, 273, 408, 409] , [311, 416, 408, 273] , [309, 293, 408, 416] , [184,  63,  79, 192] 
]

class face_image_to_face_mesh:
    def demo(self):
        demo = gr.Blocks()
        with demo:
            gr.Markdown(
            """
            # Face Image to Face Quad Mesh
            Uses MediaPipe to detect a face in an image and convert it to a (mostly) quad mesh.
            Currently saves to OBJ, hopefully glb at some point with color data.
            The 3d viewer has Y pointing the opposite direction from Blender, so ya hafta spin it.
            """)

            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(label="Input image", type="numpy", source="upload")
                    gr.Examples( examples=[
                        'examples/blonde-00019-1421846474.png',
                        'examples/dude-00110-1227390728.png',
                        'examples/granny-00056-1867315302.png',
                        'examples/tuffie-00039-499759385.png',
                    ], inputs=[upload_image] )
                    upload_image_btn = gr.Button(value="Detect faces")
                    with gr.Group():
                        min_detection_confidence = gr.Slider(label="Min detection confidence", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                        gr.Textbox(show_label=False, value="Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful.")
                with gr.Column():
                    with gr.Group():
                        num_faces_detected = gr.Number(label="Number of faces detected", value=0)
                        output_mesh = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model")
                        output_image = gr.Image(label="Output image")

            outputs = [output_mesh, output_image, num_faces_detected]
            upload_image_btn.click(
                fn=self.detect, 
                inputs=[upload_image, min_detection_confidence], 
                outputs=outputs
            )
        demo.launch()


    def detect(self, image, min_detection_confidence):
        width  = image.shape[1]
        height = image.shape[0]
        ratio  = width / height
            
        mesh = "examples/jackiechan.obj"

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return mesh, image, 0

            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mesh = self.toObj(ratio=ratio, landmark_list=face_landmarks)

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

            return mesh, annotated_image,1

    def toObj( self, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList):
        print( f'you have such pretty hair' )
        lines = []
        points = self.landmarksToPoints( ratio, landmark_list )
        for point in points:
            vertex = "v " + " ".join([str(value) for value in point])
            lines.append( vertex )
        for quad in QUADS:
            face = "f " + " ".join([str(vertex) for vertex in quad])
            lines.append( face )
            normal = self.totallyNormal( points[ quad[ 0 ] -1 ], points[ quad[ 1 ] -1 ], points[ quad[ 2 ] -1 ] )
            lines.append( "vn " + " ".join([str(value) for value in normal]) )

        obj_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        output_file = obj_file.name
        out = open( output_file, 'w' )
        out.write( '\n'.join( lines ) )
        out.close()
        print( f'I know it is special to you so I saved it to {output_file} since we are friends' )
        return output_file

    def landmarksToPoints( self, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList ):
        points = []
        mins = [+np.inf] * 3
        maxs = [-np.inf] * 3
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                    idk_what_to_do_for_this = True
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
        return points

    def totallyNormal(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        return normal.tolist()    


face_image_to_face_mesh().demo()