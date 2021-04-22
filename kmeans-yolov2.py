import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "../dataset/wider_train_annotations"

CLUSTERS = 5
WIDTH = 224
HEIGHT = 224
INPUT_SIZE = 224 * 2
STRIP_SIZE = 32


grid_w = INPUT_SIZE/STRIP_SIZE
grid_h = INPUT_SIZE/STRIP_SIZE

def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        img_width = int(tree.findtext("./size/width"))
        img_height = int(tree.findtext("./size/height"))

        cell_w = img_width/grid_w
        cell_h = img_height/grid_h

        i = 0
        for obj in tree.iter("object"):
            i += 1
            xmin = int(obj.findtext("bndbox/xmin"))*1.0 / cell_w
            ymin = int(obj.findtext("bndbox/ymin"))*1.0 / cell_h
            xmax = int(obj.findtext("bndbox/xmax"))*1.0 / cell_w
            ymax = int(obj.findtext("bndbox/ymax"))*1.0 / cell_h

            if xmin==xmax or ymin==ymax:
                print("you need to check obj[{}] {}".format(i, xml_file))
                continue

            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
