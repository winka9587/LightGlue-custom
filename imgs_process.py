import os
import os
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", 
                    default='/data4/cxx/dataset/desk_2/depth/02292.png', 
                    help="path to input directory")
args = parser.parse_args()

image_extensions = [".jpg", ".png"]
image_count = 0

# 获取图像数量
for file_name in os.listdir(args.input):
    if os.path.splitext(file_name)[1] in image_extensions:
        image_count += 1

print(f"Number of images in {args.input}: {image_count}")

for i in range(image_count):
    old_name = os.path.join(args.input, f"{i}.png")
    new_name = os.path.join(args.input, f"{i+1:05}.png")
    os.rename(old_name, new_name)
    print("rename from {} to {}".format(old_name, new_name))
