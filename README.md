# LOD_PDR

Official repository of "LiDAR-based All-weather 3D Object Detection via Prompting and Distilling 4D Radar", ECCV2024. [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07349.pdf)

## Requirements
The code has been tested with
- python 3.8
- CUDA 11.1
- pytorch 1.10.1 
- spconv-cu111 2.1.25
- open3d 0.15.2
- opencv-python 4.8.1.78
- matplotlib 3.5.3
- numba 0.53.0
- nms 0.1.6

## Usage

### Download logs
https://drive.google.com/drive/folders/13JRwnTY_cupN13-gwpXWlfIsLmIAIaRZ?usp=sharing

### Teacher
```
# Train
python main_train_0.py

# Test
python main_cond_0.py
```

### Student
```
# Train
python main_train_student.py

# Test
python main_cond_student.py
```

## Citation

If you find our work helpful, please consider citing our paper:
```
@inproceedings{Chae_2024_ECCV,
  title={LiDAR-Based All-Weather 3D Object Detection via Prompting and Distilling 4D Radar},
  author={Chae, Yujeong and Kim, Hyeonseong and Oh, Changgyoon and Kim, Minseok and Yoon, Kuk-Jin},
  booktitle={European Conference on Computer Vision},
  pages={368--385},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements

This work is developed based on the [K-Radar dataset and codebase](https://github.com/kaist-avelab/K-Radar).
