import urllib
from http import cookiejar
from urllib import request, parse
from googleapiclient.errors import HttpError
import re
from multiprocessing import Pool
from io import BytesIO
from PIL import Image
import json
import _pickle as cPickle
import numpy as np
from googleapiclient.discovery import build
import time
import sys
import argparse
import tensorflow as tf

FLAGS = None

def get_min_size(size):
    for i in range(0, 10):
        if size >= i*100 and size<=((i+1)*100)-1:
            return i*100

def get_max_size(size):
    for i in range(0, 10):
        if size >= i*100 and size<=((i+1)*100)-1:
            return ((i+1)*100)-1

def craw(_):
    multiple_results = [pool.apply_async(get_fish, [i, FLAGS]) for i in range(100*FLAGS.group_index, 100*(FLAGS.group_index+1))]
    pool.close()
    pool.join()

    for res in multiple_results:
        fish_id, genusname, speciesname, _images, _labels, _test_images, _test_labels = res.get()
        if genusname and speciesname:
            fish_map[int(fish_id)] = dict(
                genusname = genusname,
                speciesname = speciesname
            )
            images.extend(_images)
            labels.extend(_labels)
            test_images.extend(_test_images)
            test_labels.extend(_test_labels)


    max_id = 0
    min_id = 40000
    for key in fish_map:
        if key > max_id: max_id = key
        if key < min_id: min_id = key
    min_id = get_min_size(min_id)
    max_id = get_max_size(max_id)


    fish_map["max_id"] = max_id
    fish_map["min_id"] = min_id

    with open(fish_map_file + str(FLAGS.group_index) + ".json", 'w') as outfile:
        json.dump(fish_map, outfile, indent=4)

    final_labels = []
    for label in labels:
        label_list = [0] * (max_id - min_id + 1)
        label_list[label-min_id] = 1
        final_labels.append(label_list)

    final_test_labels = []
    for test_label in test_labels:
        test_label_list = [0] * (max_id - min_id + 1)
        test_label_list[test_label-min_id] = 1
        final_test_labels.append(test_label_list)


    print("Everything Completed Dumping")
    with open(data_file + str(FLAGS.group_index) + ".p" , 'wb') as datafile:
        cPickle.dump([images, final_labels, test_images, final_test_labels], datafile)


def get_fish(fish_id, FLAGS):
    genusname = None
    speciesname = None
    _images = list()
    _labels = list()
    for i in range(-1, 100):
        fish_page = urllib.request.urlopen("http://www.fishbase.org/photos/PicturesSummary.php?StartRow=" + str(i) + "&ID=" +
            str(fish_id) + "&what=species&TotRec=20")
        fish_page_content = str(fish_page.read())

        if not genusname and not speciesname:
            fish_info = re.findall(r'genusname=.+?&speciesname=.+?\\\'', fish_page_content)
            if fish_info:
                fish_info = fish_info[0]
                break_index = fish_info.find("&")
                genusname = fish_info[10:break_index]
                speciesname = fish_info[break_index+13:-2]
                print("Mapped Fish-%s-%s To %d" % (genusname, speciesname, fish_id))

        img_res = re.findall(r'<img src=\\\'.+?\\\'.+?>', fish_page_content)
        if img_res:
            img = img_res[0]
            img = img[11: img.find('alt')-3]
            print(img)
            image_array = get_image("http://www.fishbase.org/photos/" + str(img))
            if image_array not in _images:
                _images.append(image_array)
                _labels.append(fish_id)
        else:
            break
    if genusname and speciesname:
        counter = 1
        while(len(_images) < FLAGS.image_count):
            google_img_list = google_image(genusname, speciesname, counter, FLAGS)
            if google_img_list is None:
                break
            counter += 10
            for google_img_array in google_img_list:
                if google_img_array not in _images:
                    _images.append(google_img_array)
                    _labels.append(fish_id)

    if len(_images) > 5:
        return fish_id, genusname, speciesname, _images[5:], _labels[5:], _images[:5], _labels[:5]
    else:
        return fish_id, genusname, speciesname, _images, _labels, list(), list()


def resize(img):
    img.thumbnail((96, 64))
    width, height = img.size
    new_img = Image.new("RGB", [96, 64], "white")
    new_img.paste(img, (int((96 - width) / 2), int((64-height)/2)))
    return new_img

def get_image(url):
    fish_image_page = urllib.request.urlopen(url)
    img = Image.open(BytesIO(fish_image_page.read()))
    img = resize(img)
    img = img.convert('RGB')
    # img.save("test1.bmp")
    return img

def google_image(genusname, speciesname, start, FLAGS):
    res = list()
    print("Googling Image for: " + genusname+ " " + speciesname)
    results = google_search(genusname+ " " + speciesname, FLAGS, num=10, start = start)

    if results is None:
        return None

    for result in results:
        result_link = result['image']['thumbnailLink']
        print(result_link)
        res.append(get_image(result_link))

    return res

def google_search(search_term, FLAGS, **kwargs):
    try:
        service = build("customsearch", "v1", developerKey=FLAGS.api_key)
        res = service.cse().list(q=search_term, cx=FLAGS.cse_id, searchType='image', **kwargs).execute()
        return res['items']
    except HttpError as e:
        print(e)
        return None




if __name__ == "__main__":
    cj = cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    urllib.request.install_opener(opener)
    fish_map_file = "fishMap"
    data_file = "rawdata"
    pool = Pool(processes=40)
    fish_map = dict()
    images = list()
    labels = list()
    test_images = list()
    test_labels = list()



    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_count',
        type=int,
        default=200,
        help='Minimum Image Count for Each Fish')

    parser.add_argument(
        '--cse_id',
        type=str,
        default="your-search-engine-id",
        help='Google Search Engine Id')

    parser.add_argument(
        '--api_key',
        type=str,
        default="your-api-key",
        help='Google Api Key')

    parser.add_argument(
        '--group_index',
        type=str,
        default=0,
        help='Fish group index')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=craw, argv=[sys.argv[0]] + unparsed)
