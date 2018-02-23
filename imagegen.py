import _pickle as cPickle
import json
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def load_data(datafile, mapfile):
    with open(datafile, 'rb') as pfile:
        data= cPickle.load(pfile)

    with open(mapfile, 'r') as mfile:
        fish_map = json.load(mfile)

    return data, fish_map



if __name__ == "__main__":
    print("Loading Data")
    data, fish_map = load_data("rawdata0.p", "fishMap0.json")

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    new_train_image = list()
    new_train_label = list()

    for n in range(0, len(data[0])):
        x = img_to_array(data[0][n])
        x = x.reshape((1,) + x.shape)
        print(n)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            batch = batch.reshape((64,96,3))
            batch = array_to_img(batch)
            new_train_image.append(batch)
            new_train_label.append(data[1][n])
            i += 1
            if i > 10:
                break

    data[0].extend(new_train_image)
    data[1].extend(new_train_label)

    print("Saving Train Image")
    for n in range (0, len(data[0])):
        print(n)
        data[0][n].save("FishTrainImage/" + str(n) + ".jpg")

    print("Saving Test Image")
    for n in range (0, len(data[2])):
        print(n)
        data[2][n].save("FishTestImage/" + str(n) + ".jpg")

    print("Everything Completed Dumping")
    with open("data0.p" , 'wb') as datafile:
        cPickle.dump(data, datafile)
