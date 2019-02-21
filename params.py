import argparse


parser = argparse.ArgumentParser()

arg = parser.add_argument

arg('--images_dir',default='./images')
arg('--masks_dir',default='./masks')
args('--resize_size',type=int,default=160)
args('--batch_size',type=int,default=24)
args('--epochs',type=int,default=100)
args('--input_size',type=int,default=192)
args('--resize_size',type=int,default=160)


args = parser.parse_args()
