import argparse
import concurrent
import numpy as np
import vispy.scene
from vispy.scene import visuals
import h5py

from dataset.provider import S3DIS_CLASSES, VKITTI3D_CLASSES


def concat_data(arr):
    return np.concatenate(arr, axis=0)


def stack_data(arr):
    return np.row_stack(arr)


def get_label(text):
    label = vispy.scene.Label(text=text, color='white')
    label.height_max = 30
    return label


parser = argparse.ArgumentParser(allow_abbrev=False, description='Visualization script.')
parser.add_argument('--file',
                    action='store',
                    type=str,
                    help='Path of raw point cloud file.')
parser.add_argument('--semantic_classes',
                    action='store',
                    type=str,
                    default='vkitti3d',
                    help='Semantic class names and color schema [ vkitti3d | s3dis ].')

config = parser.parse_args()


data = h5py.File(config.file, 'r')

points = data['points']
npoints = data['normalized_points']
ppoints = data['predicted_points']
labels = data['labels']


with concurrent.futures.ThreadPoolExecutor() as executor:
    data_result = executor.submit(stack_data, points)
    ndata_result = executor.submit(stack_data, npoints)
    label_result = executor.submit(concat_data, labels)
    preds_result = executor.submit(stack_data, ppoints)

data = data_result.result()
ndata = ndata_result.result()
label = label_result.result()
preds = preds_result.result()

rgb_codes = [rgb_code for (_, rgb_code) in VKITTI3D_CLASSES]
class_names = [class_name for (class_name, _) in VKITTI3D_CLASSES]

if 's3dis' == config.semantic_classes:
    rgb_codes = [rgb_code for (_, rgb_code) in S3DIS_CLASSES]
    class_names = [class_name for (class_name, _) in S3DIS_CLASSES]

real_colors = np.zeros((label.shape[0], 3))
mask_colors = np.zeros((label.shape[0], 3))
pred_colors = np.zeros((label.shape[0], 3))
for i in range(data.shape[0]):
    real_colors[i, :] = [code for code in ndata[i, 3:6]]
    mask_colors[i, :] = [code / 255 for code in rgb_codes[label[i]]]
    pred_colors[i, :] = [code / 255 for code in rgb_codes[np.argmax(preds[i])]]


canvas = vispy.scene.SceneCanvas(title='Tiny-PointNet Visualization', keys='interactive', show=True, bgcolor='grey')
canvas.size = 1920, 1080
canvas.show()

vb1 = vispy.scene.widgets.ViewBox(border_color='black', parent=canvas.scene)
vb2 = vispy.scene.widgets.ViewBox(border_color='black', parent=canvas.scene)
vb3 = vispy.scene.widgets.ViewBox(border_color='black', parent=canvas.scene)
vb4 = vispy.scene.widgets.ViewBox(border_color='black', parent=canvas.scene)

grid = canvas.central_widget.add_grid()
grid.padding = 6

grid.add_widget(get_label('Real colors'), 0, 0)
grid.add_widget(get_label('Ground thrith'), 0, 1)
grid.add_widget(get_label('Prediction'), 0, 2)
grid.add_widget(vb1, 1, 0)
grid.add_widget(vb2, 1, 1)
grid.add_widget(vb3, 1, 2)

c_grid = vb4.add_grid()

c_grid.add_widget(get_label('Classes:'))

for i in range(len(rgb_codes)):
    vb = vispy.scene.widgets.ViewBox(bgcolor=[code / 255 for code in rgb_codes[i]], parent=vb4)
    vb.add_widget(get_label(class_names[i]))
    c_grid.add_widget(vb)

grid.add_widget(vb4, 2, 0, col_span=3)

real_scatter = visuals.Markers()
mask_scatter = visuals.Markers()
pred_scatter = visuals.Markers()

real_scatter.set_data(data[:, :3], edge_color=None, face_color=real_colors, size=5)
mask_scatter.set_data(data[:, :3], edge_color=None, face_color=mask_colors, size=5)
pred_scatter.set_data(data[:, :3], edge_color=None, face_color=pred_colors, size=5)

vb1.add(real_scatter)
vb2.add(mask_scatter)
vb3.add(pred_scatter)

vb1.camera = 'turntable'
vb2.camera = 'turntable'
vb3.camera = 'turntable'

vb1.camera.link(vb2.camera)
vb1.camera.link(vb3.camera)

vb1.camera.set_range()

# add a colored 3D axis for orientation
real_axis = visuals.XYZAxis(parent=vb1.scene)
mask_axis = visuals.XYZAxis(parent=vb2.scene)
pred_axis = visuals.XYZAxis(parent=vb3.scene)


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
