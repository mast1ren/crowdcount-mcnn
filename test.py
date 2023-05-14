import os
import torch
import numpy as np
import sys

sys.path.append('./src')
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils

from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio


def get_seq_class(seq, set):
    backlight = [
        'DJI_0021',
        'DJI_0022',
        'DJI_0032',
        'DJI_0202',
        'DJI_0339',
        'DJI_0340',
        'DJI_0463',
        'DJI_0003',
    ]

    fly = [
        'DJI_0177',
        'DJI_0174',
        'DJI_0022',
        'DJI_0180',
        'DJI_0181',
        'DJI_0200',
        'DJI_0544',
        'DJI_0012',
        'DJI_0178',
        'DJI_0343',
        'DJI_0185',
        'DJI_0195',
        'DJI_0996',
        'DJI_0977',
        'DJI_0945',
        'DJI_0946',
        'DJI_0091',
        'DJI_0442',
        'DJI_0466',
        'DJI_0459',
        'DJI_0464',
    ]

    angle_90 = [
        'DJI_0179',
        'DJI_0186',
        'DJI_0189',
        'DJI_0191',
        'DJI_0196',
        'DJI_0190',
        'DJI_0070',
        'DJI_0091',
    ]

    mid_size = [
        'DJI_0012',
        'DJI_0013',
        'DJI_0014',
        'DJI_0021',
        'DJI_0022',
        'DJI_0026',
        'DJI_0028',
        'DJI_0028',
        'DJI_0030',
        'DJI_0028',
        'DJI_0030',
        'DJI_0034',
        'DJI_0200',
        'DJI_0544',
        'DJI_0463',
        'DJI_0001',
        'DJI_0149',
    ]
    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    # elif seq in cloudy:
    #     light = 'cloudy'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'

    count = 'sparse'
    loca = sio.loadmat(
        os.path.join(
            '../../nas-public-linkdata/ds/dronebird/',
            set,
            'ground_truth',
            'GT_img' + str(seq[-3:]) + '000.mat',
        )
    )['locations']
    if loca.shape[0] > 150:
        count = 'crowded'
    # return light, resolution, count
    return light, angle, bird, size, count


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

data_path = './data/original/shanghaitech/part_B_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
model_path = (
    '../../nas-public-linkdata/ds/result/ccm/saved_models/mcnn_dronebirds_14.h5'
)

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir, 'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:1")
net = CrowdCounter()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.to(device)
net.eval()
mae = 0.0
mse = 0.0

# load test data
data_loader = ImageDataLoader(
    os.path.join('../../nas-public-linkdata/ds/dronebird', 'test.json'),
    shuffle=False,
    gt_downsample=True,
    pre_load=False,
)
preds = [[] for i in range(10)]
gts = [[] for i in range(10)]
i = 0
for blob in data_loader:
    im_data = blob['data']
    gt_data = blob['gt_density']
    img_path = blob['fname']
    seq = int(os.path.basename(img_path)[3:6])
    seq = 'DJI_' + str(seq).zfill(4)
    light, angle, bird, size, count = get_seq_class(seq, 'test')

    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    # count = 'crowded' if gt_count > 150 else 'sparse'
    pred_e = et_count
    gt_e = gt_count
    if light == 'sunny':
        preds[0].append(pred_e)
        gts[0].append(gt_e)
    elif light == 'backlight':
        preds[1].append(pred_e)
        gts[1].append(gt_e)
    if count == 'crowded':
        preds[2].append(pred_e)
        gts[2].append(gt_e)
    else:
        preds[3].append(pred_e)
        gts[3].append(gt_e)
    if angle == '60':
        preds[4].append(pred_e)
        gts[4].append(gt_e)
    else:
        preds[5].append(pred_e)
        gts[5].append(gt_e)
    if bird == 'stand':
        preds[6].append(pred_e)
        gts[6].append(gt_e)
    else:
        preds[7].append(pred_e)
        gts[7].append(gt_e)
    if size == 'small':
        preds[8].append(pred_e)
        gts[8].append(gt_e)
    else:
        preds[9].append(pred_e)
        gts[9].append(gt_e)

    mae += abs(gt_count - et_count)
    mse += (gt_count - et_count) * (gt_count - et_count)
    print(
        '\r[{:>{}}/{}] img: {}, error: {}, gt: {}, pred: {}'.format(
            i,
            len(str(data_loader.num_samples)),
            data_loader.num_samples,
            os.path.basename(img_path),
            abs(gt_count - et_count),
            gt_e,
            pred_e,
        ),
        end='',
    )
    i += 1
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(
            density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png'
        )
print()
mae = mae / data_loader.get_num_samples()
mse = np.sqrt(mse / data_loader.get_num_samples())
with open(file_results, 'w') as f:
    print('MAE: {:0.2f}, MSE: {:0.2f}'.format(mae, mse))
    f.write('MAE: {:0.2f}, MSE: {:0.2f}\n'.format(mae, mse))

    # f = open(file_results, 'w')
    # f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
    # f.close()
    attri = [
        'sunny',
        'backlight',
        'crowded',
        'sparse',
        '60',
        '90',
        'stand',
        'fly',
        'small',
        'mid',
    ]
    for i in range(10):
        if len(preds[i]) == 0:
            continue
        print(
            '{}: MAE:{}. RMSE:{}.'.format(
                attri[i],
                mean_absolute_error(preds[i], gts[i]),
                np.sqrt(mean_squared_error(preds[i], gts[i])),
            )
        )
        f.write(
            '{}: MAE:{}. RMSE:{}.\n'.format(
                attri[i],
                mean_absolute_error(preds[i], gts[i]),
                np.sqrt(mean_squared_error(preds[i], gts[i])),
            )
        )
