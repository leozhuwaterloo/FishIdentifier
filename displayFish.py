import _pickle as cPickle
import json
from PIL import Image
import math

def load_data(datafile, mapfile):
    with open(datafile, 'rb') as pfile:
        data= cPickle.load(pfile)

    with open(mapfile, 'r') as mfile:
        fish_map = json.load(mfile)

    return data, fish_map

if __name__ == "__main__":
    group = True
    test = True
    if not group:
        print("Loading Data")
        data, fish_map = load_data("data0.p", "fishMap0.json")
        for i in range(0, 100):
            print(i)
            data[2][i*3].save("FishImageTest/test" + str(i) + ".png")
    else:
        print("Loading Data")
        index = 2 if test else 0
        data, fish_map = load_data("data0.p", "fishMap0.json")
        start = 0
        max_count = len(data[index])
        size = int(math.ceil(math.sqrt(max_count)))
        img = Image.new("RGB", [96*size, 64*size], "white")

        for i in range(0, size):
            for j in range(0, size):
                if start + size*i + j >= max_count:
                    break
                img.paste(data[index][start + size*i + j], (96*j, 64*i))

        if test:
            img.save("TestFishCollection.png")
        else:
            img.save("TrainFishCollection.png")
