import os 
from PIL import Image
from io import BytesIO
import base64
import argparse
import csv
import random
import uuid

# uniq_id img_id  ground truth caption                           garbage (pred obj labels)     img base64 string                
# 162365  12455   the sun sets over the trees beyond some docks.  sky&&water&&dock&&pole  /9j/4AAQSkZJ....UCP/2Q==

def img2base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def write_one_image_per_task(a_e, args, csv_writer):
    a, e = a_e.split("+")
    image_fnames = os.listdir(os.path.join(args.dataset_root, a_e, 'positive'))
    image_idx = random.randint(0, len(image_fnames) - 1)
    chosen_img = image_fnames[image_idx]
    img = Image.open(os.path.join(args.dataset_root, a_e, 'positive', chosen_img))

    uniq_id = a_e + "_" + str(uuid.uuid1())
    img_id = uniq_id + "_" + str(image_idx)
    caption = f"{a} {e}"
    garbage = "sky&&water&&dock&&pole" # Not used
    img64 = img2base64(img)

    csv_writer.writerow([uniq_id, img_id, caption, garbage, img64])

def write_all_images_per_task(a_e, args, csv_writer):
    a, e = a_e.split("+")
    garbage = "sky&&water&&dock&&pole" # Not used
    caption = f"{a} {e}"

    for i, image_fname in enumerate(os.listdir(os.path.join(args.dataset_root, a_e, 'positive'))): 
        uniq_id = a_e + "_" + str(uuid.uuid1())
        img_id = a_e + "_" + image_fname
        img = Image.open(os.path.join(args.dataset_root, a_e, 'positive', image_fname))
        img64 = img2base64(img)

        csv_writer.writerow([uniq_id, img_id, caption, garbage, img64])


def main(args):
    # Open tsv file
    csv_fptr = open(args.tsv_out, 'w')
    csv_writer = csv.writer(csv_fptr, delimiter='\t')

    # for each folder in action effect dataset
        # get base64 img 
        # write line to tsv 

    for a_e in os.listdir(args.dataset_root):
        if args.mode == "one":
            write_one_image_per_task(a_e, args, csv_writer)
        else:
            write_all_images_per_task(a_e, args, csv_writer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, help="folder containing action+effeect pairs")
    parser.add_argument("--tsv-out", type=str, help="tsv to save data to")
    parser.add_argument("--mode", type=str, help="Choose from [all or one]: whether to pick one image for each ae pair, or create samples for all")
    args= parser.parse_args()
    main(args)    

