import os
import numpy as np
import PIL
from PIL import Image
import cv2
import argparse

from tqdm import tqdm

def test():
    frame_pth = '/data/hzy/projects/video_tools/face_align/ffhq_align_scripts/119/000000.jpg'
    aligned_face_path = '/data/hzy/projects/video_tools/face_align/ffhq_align_scripts/119_aligned/000000.jpg'
    crop_pth = '/data/hzy/projects/video_tools/face_align/ffhq_align_scripts/119_aligned/000000.npy'
    quad_pth = '/data/hzy/projects/video_tools/face_align/ffhq_align_scripts/119_aligned/000000_quad.npy'

    crop = np.load(crop_pth)
    print(crop)

    quad = np.load(quad_pth)
    print(quad)

    x0, y0, x1, y1 = crop
    l1 = x1-x0
    l2 = y1-y0


    frame = cv2.imread(frame_pth)
    # frame = Image.open(frame_pth)
    aligned_face = cv2.imread(aligned_face_path)
    # aligned_face = Image.open(aligned_face_path)

    print(l1, l2)

    # aligned_face = aligned_face.resize((l1,l2))
    aligned_face = cv2.resize(aligned_face, (l1, l2))

    # src_four_p = np.array([
    #         [x0,y0],
    #         [x0,y1],
    #         [x1,y1],
    #         [x1,y0]
    #         ], dtype='float32')

    # src_four_p = np.array([
    #         [x0,y0],
    #         [x0,y1],
    #         [x1,y1],
    #         [x1,y0]
    #         ], dtype='float32')
    # print(src_four_p)

    src_four_p = np.array([
            [0,0],
            [0,l2],
            [l1,l2],
            [l1,0]
            ], dtype='float32')

    dst_four_p = np.array(quad, dtype='float32')
    dst_four_p += (x0, y0)

    print(dst_four_p)

    # dst_four_p = np.array([
    #         [ 20.94571936,  19.45475454],
    #         [ 18.80475454, 202.75428064],
    #         [202.10428064, 204.89524546],
    #         [204.24524546,  21.59571936]
    #         ], dtype='float32')

    PerTransform = cv2.getPerspectiveTransform(src_four_p, dst_four_p)
    print(PerTransform)
    print(aligned_face.shape)
    inwarp_face = cv2.warpPerspective(aligned_face, PerTransform, (frame.shape[1], frame.shape[0]))

    # frame[y0:y1,x0:x1] = aligned_face

    mask = (inwarp_face.mean(2)==0).astype(dtype='float32')
    print(mask.shape)

    kernel = np.ones((3,3), np.uint8)
    
    kernel = np.ones((40,40), np.uint8)
    # mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask , kernel)
    mask = mask[:,:,np.newaxis]
    print(mask.shape)

    frame = frame * mask + inwarp_face * (1-mask)


    cv2.imwrite('./out.jpg', frame)

    # frame.paste(aligned_face, (x0, y0), mask=None)
    # frame.save('./out.jpg')

def main_before():
    parser = argparse.ArgumentParser(description="Align face images to FFHQ format")
    parser.add_argument("--frame_dir", type=str, help="path of the input image")
    parser.add_argument("--face_dir", type=str, default=None, help="output path of the aligned face")
    parser.add_argument("--aligned_dir", type=str, default=None, help="output path of the aligned face")
    parser.add_argument("--output_dir", type=str, default=None, help="output path of the aligned face")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for frame_name in tqdm(os.listdir(args.frame_dir)):
        frame_pth = os.path.join(args.frame_dir, frame_name)
        aligned_face_pth = os.path.join(args.face_dir, frame_name[:-4]+'.png')
        crop_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'.npy')
        quad_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'_quad.npy')
        save_pth = os.path.join(args.output_dir, frame_name)

        crop = np.load(crop_pth)

        quad = np.load(quad_pth)

        x0, y0, x1, y1 = crop
        l1 = x1-x0
        l2 = y1-y0

        frame = cv2.imread(frame_pth)
        
        aligned_face = cv2.imread(aligned_face_pth)
        aligned_face = cv2.resize(aligned_face, (l1, l2))

        src_four_p = np.array([
                [0,0],
                [0,l2],
                [l1,l2],
                [l1,0]
                ], dtype='float32')

        dst_four_p = np.array(quad, dtype='float32')

        PerTransform = cv2.getPerspectiveTransform(src_four_p, dst_four_p)
        aligned_face = cv2.warpPerspective(aligned_face, PerTransform, (l1,l2))

        frame[y0:y1,x0:x1] = aligned_face
        cv2.imwrite(save_pth, frame)

def main():
    parser = argparse.ArgumentParser(description="Align face images to FFHQ format")
    parser.add_argument("--frame_dir", type=str, help="path of the input image")
    parser.add_argument("--face_dir", type=str, default=None, help="output path of the aligned face")
    parser.add_argument("--aligned_dir", type=str, default=None, help="output path of the aligned face")
    parser.add_argument("--output_dir", type=str, default=None, help="output path of the aligned face")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for frame_name in tqdm(os.listdir(args.frame_dir)):
        frame_pth = os.path.join(args.frame_dir, frame_name)
        # aligned_face_pth = os.path.join(args.face_dir, frame_name[:-4]+'.png')
        aligned_face_pth = os.path.join(args.face_dir, frame_name[:-4]+'.jpg')
        crop_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'.npy')
        quad_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'_quad.npy')
        pad_pth = os.path.join(args.aligned_dir, frame_name[:-4]+'_pad.npy')
        save_pth = os.path.join(args.output_dir, frame_name)

        crop = np.load(crop_pth)

        quad = np.load(quad_pth)

        if_pad=False
        if os.path.exists(pad_pth):
                if_pad=True
                pad = np.load(pad_pth)

        x0, y0, x1, y1 = crop
        l1 = x1-x0
        l2 = y1-y0

        frame = cv2.imread(frame_pth)
        
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