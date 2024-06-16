# Plural forms of objects
import argparse
import json
import os
import random
import inflect

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output_directory', type=str, default='dataset',
                    help='The path to the dataset directory')
parser.add_argument('--no_scene_percent', type=float, default=0.5,
                    help='Percentage of no-scene samples')
parser.add_argument('--N_samples', type=int, default=200,
                    help='Number of samples to process')


coco_objs = {
    "person": [{"id": 0, "name": "person"}], 
    "vehicle": [{"id": 1, "name": "bicycle"}, {"id": 2, "name": "car"}, {"id": 3, "name": "motorcycle"}, {"id": 4, "name": "airplane"}, {"id": 5, "name": "bus"}, {"id": 6, "name": "train"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "boat"}], 
    "outdoor": [{"id": 9, "name": "traffic light"}, {"id": 10, "name": "fire hydrant"}, {"id": 11, "name": "stop sign"}, {"id": 12, "name": "parking meter"}, {"id": 13, "name": "bench"}], 
    "animal": [{"id": 14, "name": "bird"}, {"id": 15, "name": "cat"}, {"id": 16, "name": "dog"}, {"id": 17, "name": "horse"}, {"id": 18, "name": "sheep"}, {"id": 19, "name": "cow"}, {"id": 20, "name": "elephant"}, {"id": 21, "name": "bear"}, {"id": 22, "name": "zebra"}, {"id": 23, "name": "giraffe"}],
    "accessory": [{"id": 24, "name": "backpack"}, {"id": 25, "name": "umbrella"}, {"id": 26, "name": "handbag"}, {"id": 27, "name": "tie"}, {"id": 28, "name": "suitcase"}], 
    "sports": [{"id": 29, "name": "frisbee"}, {"id": 30, "name": "skis"}, {"id": 31, "name": "snowboard"}, {"id": 32, "name": "sports ball"}, {"id": 33, "name": "kite"}, {"id": 34, "name": "baseball bat"}, {"id": 35, "name": "baseball glove"}, {"id": 36, "name": "skateboard"}, {"id": 37, "name": "surfboard"}, {"id": 38, "name": "tennis racket"}], 
    "kitchen": [{"id": 39, "name": "bottle"}, {"id": 40, "name": "wine glass"}, {"id": 41, "name": "cup"}, {"id": 42, "name": "fork"}, {"id": 43, "name": "knife"}, {"id": 44, "name": "spoon"}, {"id": 45, "name": "bowl"}], 
    "food": [{"id": 46, "name": "banana"}, {"id": 47, "name": "apple"}, {"id": 48, "name": "sandwich"}, {"id": 49, "name": "orange"}, {"id": 50, "name": "broccoli"}, {"id": 51, "name": "carrot"}, {"id": 52, "name": "hot dog"}, {"id": 53, "name": "pizza"}, {"id": 54, "name": "donut"}, {"id": 55, "name": "cake"}], 
    "furniture": [{"id": 56, "name": "chair"}, {"id": 57, "name": "couch"}, {"id": 58, "name": "potted plant"}, {"id": 59, "name": "bed"}, {"id": 60, "name": "dining table"}, {"id": 61, "name": "toilet"}], 
    "electronic": [{"id": 62, "name": "tv"}, {"id": 63, "name": "laptop"}, {"id": 64, "name": "mouse"}, {"id": 65, "name": "remote"}, {"id": 66, "name": "keyboard"}, {"id": 67, "name": "cell phone"}], 
    "appliance": [{"id": 68, "name": "microwave"}, {"id": 69, "name": "oven"}, {"id": 70, "name": "toaster"}, {"id": 71, "name": "sink"}, {"id": 72, "name": "refrigerator"}], 
    "indoor": [{"id": 73, "name": "book"}, {"id": 74, "name": "clock"}, {"id": 75, "name": "vase"}, {"id": 76, "name": "scissors"}, {"id": 77, "name": "teddy bear"}, {"id": 78, "name": "hair drier"}, {"id": 79, "name": "toothbrush"}]
}


object_list = ['car', 'airplane', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'backpack', 'tie', 'ball', 'glove', 'cup', 'bowl', 'apple', 'donut', 'phone', 'clock']
coco_object_name_dict = {'ball': 'sports ball', 'glove': 'baseball glove', 'phone': 'cell phone'}

numbers = ["two", "three", "four", "five", "seven", "ten"]
number_dict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}
scenes = ['on the grass', 'on the road', 'on the ground']

def create_shuffled_list_at_length(my_list, length=100):
    new_list = []
    while len(new_list) < length:
        new_list.extend(my_list)

    random.shuffle(new_list)
    new_list = new_list[:length]

    return new_list


if __name__ == "__main__":

    args = parser.parse_args()
    directory = args.output_directory
    no_scene_precent = args.no_scene_percent
    N_samples = args.N_samples
    p = inflect.engine()

    all_vals = [coco_objs[x] for x in coco_objs]
    all_vals = [item for sublist in all_vals for item in sublist]
    obj_to_id = {x["name"]: x["id"] for x in all_vals}
    prompts = []
    datatset_objects = create_shuffled_list_at_length(object_list, N_samples)
    dataset_numbers = create_shuffled_list_at_length(numbers, N_samples)


    for i in range(N_samples):
        obj_name = datatset_objects[i]
        coco_obj_name = obj_name if (obj_name not in coco_object_name_dict) else coco_object_name_dict[obj_name]
        obj_id = obj_to_id[coco_obj_name]

        plural_obj_name = p.plural(obj_name)
        number = dataset_numbers[i]

        if random.random() < no_scene_precent:
            scene = ""
            prompt = f"A photo of {number} {plural_obj_name}"
        else:
            scene = random.choice(scenes)
            prompt = f"A photo of {number} {plural_obj_name} {scene}"

        seed = random.randint(0, 1000000)

        prompts.append({
                "prompt": prompt,
                "object": coco_obj_name,
                "object_plural": plural_obj_name,
                "object_id": obj_id,
                "scene": scene,
                "number": number,
                "int_number": number_dict[number],
                "seed": seed
            })

        os.makedirs(directory, exist_ok=True)

        # Now proceed with saving your file
        json_file_path = directory + '/data.json'
        with open(json_file_path, 'w') as file:
            json.dump(prompts, file, indent=4)

        print(len(prompts), "prompts saved to", json_file_path)