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
        r = int(255.0 - img_list[i] * 255.0)
        g = int(255.0 - img_list[i+1] * 255.0)
        b = int(255.0 - img_list[i+2] * 255.0)
        img.putpixel((xc, yc), (r,g,b))
        xc += 1
        if xc > 95 + start_xc:
            xc = start_xc
            yc += 1


def get_id(id_list):
    for i, val in enumerate(id_list):
        if val == 1:
            return i

if __name__ == "__main__":
    test = True
    if test:
        data, fish_map = load_data("data0.p", "fishMap0.json")
        for i in range(0, 100):
            img = Image.new("RGB", [96, 64], "white")
            put_image(data[2][i*3], img, 0, 0)
            img.save("FishImageTest/test" + str(i) + ".png")
    else:
        data, fish_map = load_data("data0.p", "fishMap0.json")
        start = 0
        size = 50
        img = Image.new("RGB", [96*size, 64*size], "white")
        print(fish_map.get(str(get_id(data[1][0]) + fish_map["min_id"]), "None"))

        for i in range(0, size):
            for j in range(0, size):
                put_image(data[0][start + size*i + j], img, 96*i, 64*j)

        img.save("TrainFishCollction.png")



