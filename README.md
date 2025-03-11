# DyArtbank

This repo contains the code implementation of the paper:

*DyArtbank: Diverse artistic style transfer via pre-trained stable diffusion and dynamic style prompt Artbank*

[[arXiv](https://arxiv.org/abs/xxxx.xxxx)]

![](/images/image1.png)

### Abstract
> Artistic style transfer aims to transfer the learned style onto an arbitrary
content image. However, most existing style transfer methods
can only render consistent artistic stylized images, making it difficult
for users to get enough stylized images to enjoy. To solve this
issue, we propose a novel artistic style transfer framework called
DyArtbank, which can generate diverse and highly realistic artistic
stylized images. Specifically, we introduce a Dynamic Style Prompt
ArtBank (DSPA), a set of learnable parameters. It can learn and store
the style information from the collection of artworks, dynamically
guiding pre-trained stable diffusion to generate diverse and highly
realistic artistic stylized images. DSPA can also generate random artistic image samples
with the learned style information, providing
a new idea for data augmentation. Besides, a Key Content Feature Prompt (KCFP) module is proposed to provide sufficient content prompts for pre-trained stable diffusion to preserve the detailed structure of the input content image. Extensive
qualitative and quantitative experiments verify the effectiveness of
our proposed method.

## Usage

### Preparation
```shell
pip install -r requirements.txt
```

### Training

Place your reference images in a directory, for example `sample_images/cathedral/`, then run the following:

```shell
accelerate launch train.py \
  --train_data_dir=sample_images/cathedral/ \
  --output_dir=output
```
A more comprehensive list of command arguments is shown in `train.sh`

nohup accelerate launch train.py --train_data_dir=Artworks/monet_water-lilies-1914 --output_dir=output-monet

### Generate

Assume your checkpoint is saved at `output/final-1000.pt`.

```shell
python generate.py \
  --weights_path=output/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4
```

python generate.py --weights_path=output/final-50000.pt --output_dir=output_images --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" --n_images=50 --bsz=4 --customize_prefix="a painting of" --customize_suffix="Vangogh style"

CUDA_VISIBLE_DEVICES=1 python generate-control-file.py --weights_path=output-monet/final-50000.pt --output_dir=comparison_output_images_monet --pretrained_model_name_or_path="/home/zzj/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819" --n_images=5 --bsz=4

Generate with scaled standard deviation:
```shell
python generate.py \
  --weights_path=output/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4 \
  --std_scale=2.0
```

Generate composition of multiple prompt distributions:
```shell
python generate.py \
  --weights_path output1/final-1000.pt output2/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4 \
  --std_scale 1.0 1.0 \
  --distribution_weight 0.5 0.5
```

## Acknowledgement

We borrow code from [Hugging Face diffusers](https://arxiv.org/abs/2312.14216) and [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) ([CoOp](https://github.com/KaiyangZhou/CoOp)). We thank the authors and the open source contributors for their work and contribution.
