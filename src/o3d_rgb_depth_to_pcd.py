import cv2, os, sys
import numpy as np
import open3d as o3d

def main():

    argv = sys.argv
    argc = len(argv)

    if argc < 3:
        print('%s displays 3D model from RGB image and depth imag' % argv[0])
        print('[usage] python %s <rgb image> <depth image> [<zScale> <focal_length:x> <focal_length:y>]' % argv[0])
        quit()

    rgb = cv2.imread(argv[1])
    rgb = cv2.flip(rgb, 1)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    depth = cv2.imread(argv[2], cv2.IMREAD_UNCHANGED)
    depth = cv2.flip(depth, 1)

    fx = width
    fy = height

    zScale = 2

    if argc > 3:
        zScale = float(argv[3])
    
    if argc > 4:
        fx = int(argv[4])

    if argc > 5:
        fy = int(argv[5])

    cx = width // 2
    if argc > 6:
        cx = int(argv[6])

    cy = height // 2
    if argc > 7:
        cy = int(argv[7])

    print('zScale:%.1f, fx:%d, fy:%d, cx:%d, cy:%d' % (zScale, fx, fy, cx, cy))

    RGB = o3d.geometry.Image(rgb)

    #depth = 65535 - depth
    depth //= 2
    depth += 30000

    DEPTH = o3d.geometry.Image(depth)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            RGB, DEPTH, depth_scale=65535, convert_rgb_to_intensity=False)

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

    angle = np.arctan(width/height)
    cos = np.cos(-angle)
    sin = np.sin(-angle)

    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, zScale, 0], [0, 0, 0, 1]])
    #pcd.transform([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos * zScale, 0], [0, 0, 0, 1]])

    pcd.transform([[1, 0, 0, 0], [0, cos, 0, 0], [0, 0, cos * zScale, 0], [0, 0, 0, 1]])

    base = os.path.basename(argv[1])
    filename, _ = os.path.splitext(base)

    #dst_path = '%s_o3d.ply' % filename
    dst_path = 'o3d.ply'
    o3d.io.write_point_cloud(dst_path, pcd)
    print('save %s' % dst_path)

    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
