import os
import argparse
import h5py

from model.model import get_model


parser = argparse.ArgumentParser(allow_abbrev=False, description='Prediction script.')
parser.add_argument('--num_points',
                    action='store',
                    type=int,
                    default=4096,
                    help='Number of points.')
parser.add_argument('--num_classes',
                    action='store',
                    type=int,
                    default=13,
                    help='Number of segmentation classes.')
parser.add_argument('--model',
                    action='store',
                    type=str,
                    help='Path of pretrained model.')
parser.add_argument('--weights',
                    action='store',
                    type=str,
                    help='Path of pretrained model weights.')
parser.add_argument('--file',
                    action='store',
                    type=str,
                    help='Path of raw point cloud file.')
parser.add_argument('--output',
                    action='store',
                    type=str,
                    help='Path of prediction destination folder.')

config = parser.parse_args()


if not os.path.exists(config.output):
    os.mkdir(config.output)


model = get_model(num_points=config.num_points, num_classes=config.num_classes)

model.load_weights(config.weights)

f = h5py.File(config.file, 'r')

points = f['points']
npoints = f['normalized_points']
labels = f['labels']

data = npoints[:, :, 0:3]

predictions = model.predict(data)

print(f'file: {config.file} - prediction shape: {predictions.shape}')

pred_filename = config.output + '/prediction_' + config.file.split("/")[-1]


pf = h5py.File(pred_filename, 'w')
pf.create_dataset('points', data=points, compression='gzip', dtype='float32')
pf.create_dataset('normalized_points', data=npoints, compression='gzip', dtype='float32')
pf.create_dataset('predicted_points', data=predictions, compression='gzip', dtype='float32')
pf.create_dataset('labels', data=labels, compression='gzip', dtype='int64')

f.close()
pf.close()