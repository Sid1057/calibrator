import cv2 as cv
import numpy as np
import yaml
import weakref
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, help="Path for yor uncalibrated img")
parser.add_argument("--config", type=str, help="Path for yor yaml configuration file")
args = parser.parse_args()

print(args)

config = None
with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        print(config)
    except yaml.YAMLError as e:
        print(e)

img = cv.imread(args.img)

roi_node = {
    'roi_x': {
        'min_val': 0,
        'max_val': img.shape[1],
        'step': 1
    },
    'roi_y': {
        'min_val': 0,
        'max_val': img.shape[0],
        'step': 1
    },
    'roi_w': {
        'min_val': 0,
        'max_val': img.shape[1],
        'step': 1
    },
    'roi_h': {
        'min_val': 0,
        'max_val': img.shape[0],
        'step': 1
    },
}

K = np.zeros((3, 3), dtype=np.float32)
K[2][2] = 1
dist = np.zeros((1, 5), dtype=np.float32)
rvec = np.zeros((1, 3), dtype=np.float32)
tvec = np.zeros((1, 3), dtype=np.float32)


title_window = 'interface'

parameters = {
    'F': 0,
    'CX': 0,
    'CY': 0,

    'D0': 0,
    'D1': 0,
    'D2': 0,
    'D3': 0,
    'D4': 0,

    'X': 0,
    'Y': 0,
    'Z': 0,
    'yaw': 0,
    'pitch': 0,
    'roll': 0,

    'QX': 0,
    'QY': 0,
    'QZ': 0,
    'length_X': 0,
    'length_Y': 0,
    'length_Z': 0,

    'roi_x': 0,
    'roi_y': 0,
    'roi_w': img.shape[1],
    'roi_h': img.shape[0],
}

io = {
    'input_fname': '',
    'output_dir': ''
}

interface = {
    'show_pp': True,
    'render_cube': True,
    'show_ROI': True,
    'show_withour_ROI': True,
    'show_original': True,
    'custom_roi': True,
    'print_online': True,
    'width': 400
}

def project_cube():
    global parameters, img, io, interface, config
    dist = np.zeros((1, 5), dtype=np.float32)
    for i in range(5):
        dist[0][i] = parameters[f'D{i}']
    K = np.zeros((3, 3), dtype=np.float32)
    K[0][0] = parameters['F']
    K[1][1] = parameters['F']
    K[0][2] = parameters['CX']
    K[1][2] = parameters['CY']
    K[2][2] = 1

    rvec = np.zeros((1, 3), dtype=np.float32)
    tvec = np.zeros((1, 3), dtype=np.float32)


def create_ui():
    global parameters, img, io, interface, config

    cv.namedWindow('intrinsic')
    cv.namedWindow('extrinsic')
    cv.namedWindow('cube')

    io['output_dir'] = config['io']['output_dir']

    interface['show_pp'] = bool(config['interface']['show_pp'])
    interface['show_original'] = bool(config['interface']['show_original'])
    interface['print_online'] = bool(config['interface']['print_online'])
    interface['custom_roi'] = bool(config['interface']['custom_roi'])
    interface['width'] = int(config['interface']['width'])

    if interface['custom_roi'] is True:
        cv.namedWindow('roi')

    def create_trackbar_parameters_handler(key, step, alpha):
        def trackbar_handler(val):
            parameters[key] = val * step + alpha
            if interface['print_online']:
                print(json.dumps(parameters, sort_keys=False, indent=4))

        return trackbar_handler

    def create_trackbar(window_name, name, key_node, default_value=None):
        step = float(key_node[name]['step'])
        min_val = float(key_node[name]['min_val'])
        max_val = int((float(key_node[name]['max_val']) - min_val) / step)
        val = int(default_value / step) if default_value is not None else int( max_val / 2 )

        cv.createTrackbar(
            name, window_name, val, max_val, create_trackbar_parameters_handler( name, step, min_val ) )


    create_trackbar('intrinsic', 'F', config['K'], default_value=img.shape[1])
    for i in range(5):
        create_trackbar('intrinsic', f'D{i}', config['D'])
    create_trackbar('intrinsic', 'CX', config['K'], default_value=img.shape[1]/2)
    create_trackbar('intrinsic', 'CY', config['K'], default_value=img.shape[0]/2)


    create_trackbar('extrinsic', 'yaw', config['R'])
    create_trackbar('extrinsic', 'pitch', config['R'])
    create_trackbar('extrinsic', 'roll', config['R'])

    create_trackbar('extrinsic', 'X', config['T'])
    create_trackbar('extrinsic', 'Y', config['T'])
    create_trackbar('extrinsic', 'Z', config['T'])

    create_trackbar('roi', 'roi_x', roi_node, default_value=0)
    create_trackbar('roi', 'roi_y', roi_node, default_value=0)
    create_trackbar('roi', 'roi_w', roi_node, default_value=img.shape[1])
    create_trackbar('roi', 'roi_h', roi_node, default_value=img.shape[0])


    create_trackbar('cube', 'length_X', config['CUBE'], default_value=1)
    create_trackbar('cube', 'length_Y', config['CUBE'], default_value=1)
    create_trackbar('cube', 'length_Z', config['CUBE'], default_value=1)
    create_trackbar('cube', 'QX', config['CUBE'])
    create_trackbar('cube', 'QY', config['CUBE'])
    create_trackbar('cube', 'QZ', config['CUBE'], default_value=5)


if __name__ == "__main__":
    create_ui()

    if interface['show_original']:
        cv.imshow('original', img)
        cv.waitKey(100)

    while True:
        for i in range(5):
            dist[0][i] = parameters[f'D{i}']
        K[0][0] = parameters['F']
        K[1][1] = parameters['F']
        K[0][2] = parameters['CX']
        K[1][2] = parameters['CY']

        rvec[0][0] = parameters['yaw']
        rvec[0][1] = parameters['pitch']
        rvec[0][2] = parameters['roll']

        tvec[0][0] = parameters['X']
        tvec[0][1] = parameters['Y']
        tvec[0][2] = parameters['Z']

        dst = cv.undistort(img, K, dist)

        if interface['render_cube']:
            objpoints = np.zeros((9, 3), dtype=np.float32)
            QX, QY, QZ = parameters['QX'], parameters['QY'], parameters['QZ']
            LX, LY, LZ = parameters['length_X'], parameters['length_Y'], parameters['length_Z']

            objpoints[0] = (QX-LX/2, QY-LY, QZ-LZ/2)
            objpoints[1] = (QX-LX/2, QY, QZ-LZ/2)
            objpoints[2] = (QX+LX/2, QY, QZ-LZ/2)
            objpoints[3] = (QX+LX/2, QY-LY, QZ-LZ/2)

            objpoints[4] = (QX-LX/2, QY-LY, QZ+LZ/2)
            objpoints[5] = (QX-LX/2, QY, QZ+LZ/2)
            objpoints[6] = (QX+LX/2, QY, QZ+LZ/2)
            objpoints[7] = (QX+LX/2, QY-LY, QZ+LZ/2)

            objpoints[8] = (QX, QY, QZ)

            imgpoints, _ = cv.projectPoints(objpoints, rvec, tvec, K, dist)
            imgpoints = imgpoints.reshape(-1, 2).astype(np.int32)
            blue = (255, 0, 0)
            green = (0, 255, 0)
            red = (0, 0, 255)
            for i in range(4):
                cv.line(dst, tuple(imgpoints[4+i%4].tolist()), tuple(imgpoints[4+(i+1)%4].tolist()), green, 2)
            for i in range(4):
                cv.line(dst, tuple(imgpoints[i%4].tolist()), tuple(imgpoints[4+i%4].tolist()), red, 2)
            cv.circle(dst, tuple(imgpoints[8].tolist()), 3, (0, 0, 255), cv.FILLED)
            for i in range(4):
                cv.line(dst, tuple(imgpoints[i%4].tolist()), tuple(imgpoints[(i+1)%4].tolist()), blue, 2)

        # optimal matrix
        x, y = 0, 0
        h, w = img.shape[:2]
        K_original = K.copy()

        if interface['custom_roi']:
            x = min(max(0, parameters['roi_x']), img.shape[1]-2)
            y = min(max(0, parameters['roi_y']), img.shape[0]-2)
            w = min(max(2, parameters['roi_w']), img.shape[1])
            h = min(max(2, parameters['roi_h']), img.shape[0])
        else:
            _, roi = cv.getOptimalNewCameraMatrix(
                K, dist, (w, h), 1, (w,h))
            x, y, w, h = roi
            parameters['roi_x'] = x
            parameters['roi_y'] = y
            parameters['roi_w'] = w
            parameters['roi_h'] = h
        K[0][2] -= x
        K[1][2] -= y
        roi = np.asarray((x, y, w, h)).astype(np.int32)
        x, y, w, h = roi

        if interface['show_pp']:
            cx, cy = int(parameters['CX']), int(parameters['CY'])
            dst[cy, cx-10:cx+10] = (0, 0, 255)
            dst[cy-10:cy+10, cx] = (0, 0, 255)

        if w > 0 and h > 0 and interface['show_ROI']:
            cv.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv.imshow('result with roi', dst[y:y+h, x:x+w])

        # if size > screen size do pyrDown
        if interface['show_withour_ROI']:
            cv.imshow('result', dst)

        cv.imshow('intrinsic', np.zeros((1, int(interface['width'])), dtype=np.uint8))
        cv.imshow('extrinsic', np.zeros((1, int(interface['width'])), dtype=np.uint8))
        cv.imshow('cube', np.zeros((1, int(interface['width'])), dtype=np.uint8))

        if interface['custom_roi']:
            cv.imshow('roi', np.zeros((1, int(interface['width'])), dtype=np.uint8))

        key = cv.waitKey(10)
        if key in (ord('s'), ord('S')):
            np.savetxt(io['output_dir']+'K.txt', K)
            np.savetxt(io['output_dir']+'K_original.txt', K_original)
            np.savetxt(io['output_dir']+'D.txt', dist)
            np.savetxt(io['output_dir']+'roi.txt', roi)
            np.savetxt(io['output_dir']+'rvec.txt', rvec)
            np.savetxt(io['output_dir']+'tvec.txt', tvec)
            print('saved')
