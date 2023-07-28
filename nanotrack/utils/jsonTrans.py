import json
import os, sys

json_path = 'VOT2018.json'
json_path1 = 'VOT2018.json'
txt_path = 'list.txt'
dict = {}


def get_json_data(json_path):
    with open(json_path) as f:

        params = json.load(f)
        # add '/color/'to the path
        file = open(r"E:\SiamProject\NanoTrack\datasets\VOT2018\list.txt")
        while 1:
            lines = file.readlines(1000)
            if not lines:
                break
            for line in lines:
                line = line[:-1]
                root = (params[line]["img_names"])
                # print(len(root))
                while 1:
                    for i in range(len(root)):
                        kind, color, jpg = root[i].split("/")
                        # root[i] = kind + '/color/' + jpg
                        root[i] = kind + '/' + jpg
                    # print(root)
                    break

        file.close()
        # print("params",params)
        dict = params
    f.close()
    return dict


def write_json_data(dict):
    with open(json_path1, 'w') as r:
        json.dump(dict, r)
    r.close()


if __name__ == "__main__":
    the_revised_dict = get_json_data(r"E:\SiamProject\NanoTrack\datasets\VOT2018\VOT2018.json")

    write_json_data(the_revised_dict)