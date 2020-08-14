
import json
import cv2


def txtToJson(file_path):
    """
    读取文件并将转化为json
    :param file_path: 文件绝对路径
    :return: json
    """
    with open(file_path,'r', encoding="utf-8") as file:
        result = file.read()
        result = json.loads(result)
        print(result)
    return result,


def intv(*value):
    if len(value) == 1:
        # one param
        value = value[0]
    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)

def drawbbox(image, bbox, text='test'):
    color = [0, 255, 0]
    thickness = 2
    textcolor = (0, 0, 0)
    x, y, r, b = intv(bbox)
    w = r + x + 1
    h = b + y + 1
    cv2.rectangle(image, (x, y),( w,h), color, thickness, 16)
    cv2.putText(image, text , (x, y ), 0, 0.5, textcolor, 1, 16)
    return image

def get_image_info(json):
    images_arr = json[0]['images']
    annotations_arr = json[0]['annotations']
    categories_arr = json[0]['categories']
    if len(images_arr) and len(annotations_arr):
        for image in images_arr:
            image_id = image['id']
            image_name = image['file_name']
            for annotations in annotations_arr:
                image_id_anno =annotations['image_id']
                if(image_id == image_id_anno):
                    bbox=annotations['bbox']
                    category_id = annotations['category_id']
                    break
    if(len(categories_arr)):
        for categories in categories_arr:
            if(category_id == categories['id']):
                category_name = categories['name']
    return image_name,bbox,category_name

if __name__ == '__main__':
    # json标注文件路径
    path = "instance_val2017_sample.txt"
    json = txtToJson(path)
    image_name,bbox,category_name = get_image_info(json)
    img = cv2.imread(image_name)
    image = drawbbox(img,bbox,category_name)
    cv2.imshow('name', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
