# HKNet: An Aligned-Kernel Network for Image Dehazing

## Authors
Li Wan, Xiaolin Zhang

![arch](pic/all.png)

## Abstract

Image restoration, which aims to reconstruct high-quality visual content from degraded inputs, plays a crucial role in numerous practical applications. Although Transformer-based architectures have achieved leading performance in this field due to their powerful global modeling capabilities, their inherent quadratic computational complexity severely restricts their deployment efficiency. To address this challenge, we propose an efficient Aligned-Kernel Network, termed AKNet. The core of our network is the designed Aligned-Kernel Module, which innovatively employs ultra-large-kernel convolutions to capture a global receptive field in a cost-effective manner. This approach emulates the long-range dependency modeling of Transformers and is further augmented with strip convolutions to enhance multi-scale feature representations. Furthermore, we introduce a Frequency-aware Fusion module for efficient multi-scale feature integration. This module adaptively synergizes low-frequency components, which contain global structures, with high-frequency components that represent fine details at the frequency level, thereby significantly improving the quality of feature fusion. Extensive evaluations on ten benchmark datasets across diverse tasks, including image dehazing, desnowing, and defocus deblurring, demonstrate that our proposed AKNet outperforms existing methods on multiple metrics, achieving a superior balance between reconstruction performance, parameter count, and computational complexity. 

## Experiments

### Dehaze
![haze](pic/indoor_outdoor_Overlap.png)
### Desnow
![desnow](pic/snow.png)
### Defocus Deblur
![desnow](pic/dpdd.png)

## Datasets

### Dehazing Datasets
- [ITS](https://pan.baidu.com/s/11Pfl227viFijuw8jmAGcJw?pwd=m82m)
- [OTS](https://pan.baidu.com/s/10N63st8dlzkGB5v-JjUULg?pwd=16ag)
- [O-HAZE](https://pan.baidu.com/s/1e8OG6aNgFSm9SHYQ2pvyuA?pwd=7xwn)
- [NH-HAZE](https://pan.baidu.com/s/1seuSmqRUAgC5zGXS9xtxPw?pwd=nyyt)
- [DENSE-HAZE](https://pan.baidu.com/s/1hjysGvoVatWaY_FrmWeLmQ?pwd=xfu4)
- [SOTS](https://pan.baidu.com/s/1mcULooUYzGBRgqaEGFLQ0Q?pwd=779y)

### Defocus Deblur Datasets
- [DPDD](https://pan.baidu.com/s/1x1PPGvtmPpsxLBZtsT0cMA?pwd=ttj8)

### Desnowing Datasets
- [CSD](https://pan.baidu.com/s/1iUC3Y5Wn_rpy4P48x5hpVQ?pwd=352s)
- [SRRS](https://pan.baidu.com/s/14bGq_pvpUXv1k1wWJqZs4g?pwd=vcda)
- [Snow100K](https://pan.baidu.com/s/1TjR1VIn6MIqAD7UjguAmqg?pwd=4wi3)

## Environment Setup

### Using Conda

This project uses Conda to manage dependencies. Follow these steps to set up the environment:

1. Ensure you have Anaconda or Miniconda installed.
2. Run the following command in the project root directory to create the environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate py310
   ```

## Training

To train the model, run the following command:

```bash
python main.py --mode train --data_dir /path/to/your/dataset
```

Optional arguments:
- `--batch_size`: Batch size, default is 8
- `--learning_rate`: Learning rate, default is 2e-4
- `--num_epoch`: Number of training epochs, default is 500
- `--save_freq`: Model saving frequency (every N epochs), default is 50
- For more arguments, please refer to the argparse section in `main.py`

Training results will be saved in the `results/OKNet/ots/` directory.

## Testing

### pretrained  model

download model pth files from modelscope [AKNet](https://www.modelscope.cn/models/zhangasd/AKNet/)

To test the model, run the following command:

```bash
python main.py --mode test --data_dir /path/to/your/dataset --test_model /path/to/your/model.pkl --result_dir /path/to/save/results
```

Important arguments:
- `--test_model`: Path to the pre-trained model
- `--save_image`: Whether to save test result images, default is False

Test results will be saved in the specified `--result_dir` directory.
  
## Issues

If you encounter any issues, please open an issue on GitHub.