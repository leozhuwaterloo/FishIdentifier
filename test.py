import _pickle as cPickle
import json
from PIL import Image

def load_data(datafile, mapfile):
    with open(datafile, 'rb') as pfile:
        data= cPickle.load(pfile)

    with open(mapfile, 'r') as mfile:
        fish_map = json.load(mfile)

    return data, fish_map


def put_image(img_list, img, start_xc, start_yc):
    xc = start_xc
    yc = start_yc
    for i in range(0, len(img_list), 3):
        r = img_list[i]
        g = img_list[i+1]
        b = img_list[i+2]
        img.putpixel((xc, yc), (r,g,b))
        xc += 1
        if xc > 95 + start_xc:
            xc = start_xc
            yc += 1


if __name__ == "__main__":
    print("Loading Data")
    data, fish_map = load_data("data0.p", "fishMap0.json")
    img_list = list()
    img_array_list = list()
    for i in range(0, len(data[0])):
        print(i)
        img = Image.new("RGB", [96, 64], "white")
        put_image(data[0][i], img, 0, 0)
        img_list.append(img)
        img_array_list.append(data[0][i])

    with open("test_img.p" , 'wb') as datafile:
        cPickle.dump(img_list, datafile)

    with open("test_img_array.json" , 'w') as datafile:
        json.dump(img_array_list, datafile)
