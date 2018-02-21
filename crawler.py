import urllib
from http import cookiejar
from urllib import request, parse
import re
from multiprocessing import Pool
from io import BytesIO
from PIL import Image
import json
import _pickle as cPickle
import numpy as np
from googleapiclient.discovery import build
import time

def get_min_size(size):
    if size >= 0 and size<=99:
        return 0
    elif size >= 100 and size<=199:
        return 100

def get_max_size(size):
    if size >= 0 and size<=99:
        return 99
    elif size >= 100 and size<=199:
        return 199

def craw(group_index):
    multiple_results = [pool.apply_async(get_fish, [i]) for i in range(100*group_index, 100*(group_index+1))]
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

    with open(fish_map_file + str(group_index) + ".json", 'w') as outfile:
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

    with open(data_file + str(group_index) + ".p" , 'wb') as datafile:
        cPickle.dump([images, final_labels, test_images, final_test_labels], datafile)


def get_fish(fish_id):
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
            _images.append(get_image("http://www.fishbase.org/photos/" + str(img)))
            _labels.append(fish_id)
        else:
            break	
    if genusname and speciesname:
        counter = 1
        while(len(_images) < 50):
            google_img_list = google_image(genusname, speciesname, counter)
            counter += 10
            _images.extend(google_img_list)
            _labels.extend([fish_id] * len(google_img_list))
        
    if len(_images) > 30:
        return fish_id, genusname, speciesname, _images[5:], _labels[5:], _images[:5], _labels[:5]
    else:
        return fish_id, genusname, speciesname, _images, _labels, list(), list()

def get_image(url):
    fish_image_page = urllib.request.urlopen(url)
    img = Image.open(BytesIO(fish_image_page.read()))
    img = img.resize((96, 64))
    img = img.convert('RGB')
    # img.save("test1.bmp")
    img = np.asarray(img, dtype=np.int)
    img = img.reshape([96*64*3, 1])
    final_img = []
    for pixel in img:
        alpha = pixel[0]
        final_img.append((255.0 - alpha) / 255.0)
    return final_img

def google_image(genusname, speciesname, start):
    res = list()
    print("Googling Image for: " + genusname+ " " + speciesname)
    results = google_search(genusname+ " " + speciesname, "api_key", "cse_id", num=10, start = start)
    
    for result in results:
        result_link = result['image']['thumbnailLink']
        print(result_link)
        res.append(get_image(result_link))
    
    return res

def google_search(search_term, api_key, cse_id, **kwargs):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, searchType='image', **kwargs).execute()
        return res['items']    
    except HttpError as e:
        print(e)
        print("Sleeping for: " + search_term)
        time.sleep(100)
        return google_search(search_term, api_key, cse_id, **kwargs)

    


if __name__ == "__main__":
    cj = cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    urllib.request.install_opener(opener)
    fish_map_file = "fishMap"
    data_file = "data"
    pool = Pool(processes=30)
    fish_map = dict()
    images = list()
    labels = list()
    test_images = list()
    test_labels = list()

    craw(1)
