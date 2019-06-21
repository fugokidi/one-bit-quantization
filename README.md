# one-bit-quantization
Code for APSIPA 2019 paper

# Requirements
* sacred
* pytorch setup

# Training
`python train-exp.py` will train all 15 models

# Test
You need to specify which model to load.
The test will do both normal and adversarialy test.
You many want to write a bash script to run all tests.

### To test 1 bit model

`python train-exp.py with bit=1 dithering=Fasle`

### To test 1 bit dithered model

`python train-exp.py with bit=1 dithering=True`

### To test a model with differt epsilon

You need to specify the model name in the script `epsilon-exp.py`.
I hard coded the name of the file in the script, sorry.
