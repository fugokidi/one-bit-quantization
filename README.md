# one-bit-quantization
Code for APSIPA 2019 paper

# Requirements
* sacred
* pytorch setup

# Training
`python train-exp.py` will train all 15 models

# Test
You need to specify which model to load.
The test will do both normal and adversarial test.

### For example:
#### To test 1 bit model

`python eval-exp.py with bit=1 dithering=Fasle`

#### To test 1 bit dithered model

`python eval-exp.py with bit=1 dithering=True`
