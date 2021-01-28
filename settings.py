import os

working_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(working_dir, 'data')
weights_dir = os.path.join(working_dir, 'weights')
batch_size = 128

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
