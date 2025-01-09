import os
import numpy as np
import PIL
from PIL import Image
import cv2
import argparse

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Inverse the alignment face images to FFHQ format")
    parser.add_argument("--image_format", type=str, default='jpg')
    parser.add_argument("--frame_dir", type=str, help="path of the input image")
    parser.add_argument("--face_dir", type=str, default=None)
    parser.add_argument("--aligned_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for frame_name in tqdm(os.listdir(args.frame_dir)):
        frame_pth = os.path.join(args.frame_dir, frame_name)
        aligned_face_pth = os.path.join(args.face_dir, frame_name[:-4]+'.'+args.image_format)
        crop_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'.npy')
        quad_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'_quad.npy')
        pad_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'_pad.npy')
        save_pth = os.path.join(args.output_dir, frame_name)

        if os.path.exists(save_pth):
                continue

        frame = cv2.imread(frame_pth)

        if not os.path.exists(crop_pth):
                print(crop_pth)
                cv2.imwrite(save_pth, frame)
                continue

        crop = np.load(crop_pth)

        quad = np.load(quad_pth)

        if_pad=False
        if os.path.exists(pad_pth):
                if_pad=True
                pad = np.load(pad_pth)

        x0, y0, x1, y1 = crop
        l1 = x1-x0
        l2 = y1-y0

        
        aligned_face = cv2.imread(aligned_face_pth)
        aligned_face = cv2.resize(aligned_face, (l1, l2))

        src_four_p = np.array([
                [0,0],
                [0,l2],
                [l1,l2],
                [l1,0]
                ], dtype='float32')

        dst_four_p = np.array(quad, dtype='float32')
        # print(dst_four_p)
        dst_four_p_raw = dst_four_p.copy()
        if if_pad:
                dst_four_p -= pad[:2]
        if l1 < frame.shape[0] or l2 < frame.shape[1]:
                dst_four_p += (x0, y0)
        # print(dst_four_p)


        PerTransform = cv2.getPerspectiveTransform(src_four_p, dst_four_p)
        inwarp_face = cv2.warpPerspective(aligned_face, PerTransform, (frame.shape[1], frame.shape[0]))



        mask = (inwarp_face.mean(2)==0).astype(dtype='float32')

        kernel = np.ones((3,3), np.uint8)
        kernel = np.ones((40,40), np.uint8)
        
        mask = cv2.dilate(mask, kernel)
        # mask = cv2.erode(mask, kernel)
        
        kernel_size = (20, 20)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        mask = cv2.GaussianBlur(mask, blur_size, 0)
        
        mask = mask[:,:,np.newaxis]

        frame = frame * mask + inwarp_face * (1-mask)

        debug = False
        if debug:
                for src_p in src_four_p:
                        # print(src_p)
                        cv2.circle(frame, src_p.astype(int), radius=10, color=(0,255,0))
                for dst_p in dst_four_p_raw:
                        cv2.circle(frame, dst_p.astype(int), radius=10, color=(0, 0, 255))
                for dst_p in dst_four_p:
                        cv2.circle(frame, dst_p.astype(int), radius=10, color=(255, 0, 0))
        cv2.imwrite(save_pth, frame)

if __name__ == '__main__':
    main()