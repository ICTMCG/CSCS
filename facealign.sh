frames_dir=dir/path/to/unaligned/images
aligned_dir=dir/path/to/aligned/images

cd align
python align_faces_parallel.py \
    --input $frames_dir \
    --output $aligned_dir \
    --num_threads 64

cd ..