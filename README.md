
# Make It Count: Text-to-Image Generation with an Accurate Number of Objects
Despite the unprecedented success of text-to-image diffusion models, controlling the number of depicted objects using text is surprisingly hard. This is important for various applications from technical documents, to children’s books to illustrating cooking recipes. Generating object-correct counts is fundamentally challenging because the generative model needs to keep a sense of separate identity for every instance of the object, even if several objects look identical or overlap, and then carry out a global computation implicitly during generation. It is still unknown if such representations exist. To address count-correct generation, we first identify
features within the diffusion model that can carry the object identity information.
We then use them to separate and count instances of objects during the denoising
process and detect over-generation and under-generation. We fix the latter by
training a model that proposes the right location for missing objects, based on
the layout of existing ones, and show how it can be used to guide denoising
with correct object count. Our approach, CountGen, does not depend on external
source to determine object layout, but rather uses the prior from the diffusion
model itself, creating prompt-dependent and seed-dependent layouts. Evaluated on
two benchmark datasets, we find that CountGen strongly outperforms the count accuracy of existing baselines.


## Setup
Clone the repository and navigate into the directory:
```
git clone https://github.com/Litalby1/make-it-count.git
cd make-it-count
```

## Environment
Install necessary packages:
```
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Relayout Checkpoints & Datasets
For inference, download the checkpoints from [here](https://drive.google.com/file/d/1xyfkwmX9plMB5-c0VDwl7WiPuQ2qt5yb/view?usp=drive_link).
Additionally, the Relayout training dataset can be accessed [here](https://drive.google.com/drive/folders/1BoXN2KQZQZ7fCeD3aTMCrZByiVPzeyIM?usp=drive_link).

After downloading the weights, place them in `pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth`.
#### Datasets Creation:
1. CoCoCount Dataset:
   to generate the CoCoCount dataset, use this following command:

```
python dataset/create_data_CoCoCount.py \
--output_directory <output_path_for_json> \
--N_samples <number_of_data_samples>
```
You can also adjust the proportion of samples that contain no scenes by using the option: --no_scene_percent <p_between_0-1>.

2. Compbench Dataset:
   first, download the necessary CSV file from the provided link [here](https://drive.google.com/file/d/1Lya24Qc1D36wlcXeUZHEi5TwItQg2Hen/view?usp=drive_link).
   After downloading the CSV, use the command below to generate the dataset:
```
python dataset/create_data_compbench.py \
--output_directory <output_path_for_json> \
--compbench_csv <path_to_compbench_csv>
```
Replace <output_path_for_json> with the path where you want to save the JSON files and <path_to_compbench_csv> with the path to the downloaded CSV file.

## Run CountGen 🌠

1. Begin by downloading the necessary relayout checkpoints. Once downloaded, place them in `pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth`. Additionally, define the output_path in the same configuration file to designate where the results should be saved.
3. To run CountGen, use the following command template:
```
python pipeline/run_countgen.py \
--prompt <your_prompt> \
--seed <seed> \
--config <optional:config_path> \
--dataset_file <optional:path_to_dataset> 
```

Single Prompt Example:
```
python pipeline/run_countgen.py \
--prompt "A photo of six kittens sitting on a branch" \
--seed 1
```

Dataset Run Example:
```
python pipeline/run_countgen.py \
--prompt "A photo of six kittens sitting on a branch" \
--seed 1 \
--dataset_file "dataset/CoCoCount.json"
```

Modify the settings in pipeline/pipeline_config.yaml as needed for custom configurations.

### Run Using the Notebook
Run the notebook at `pipeline/countgen.ipynb`


## Train Relayout 📉
First, install necessary packages:
```
pip install -r train_relayout/relayout_requirements.txt
```


#### 1️⃣ Data Creation:
Generate training datasets for ReLayout:
```
python train_relayout/data_creation/generate_counting_unet_data.py \
--output_dir "path/to/your/output_directory"
```
This will save the generated data to the specified output directory. 

#### 2️⃣ Data Preperation:
To prepare data, run the following matching algorithm:

```
python train_relayout/data_creation/matching_algorithm.py
--data_dir <path_for_dataset_created_from_previous_code>
--output_dir <save_training_data_results_after_matching>
```
Ensure to apply this algorithm to both the training and test sets.

#### 3️⃣ Training:
Configure the paths in train_relayout/unet_config.yaml:
1. train_data_dir: ["<path_to_train_data_dir_after_matching_algorithm>"]
2. test_data_dir: ["<path_to_test_data_dir_after_matching_algorithm>"]

You may run multiple files, for example: ["<path_to_train_dir_1>, "<path_to_train_dir_2>"]
Adjust other settings and hyperparameters in train_relayout/unet_config.yaml as needed. Start the training process:

```
python train_relayout/train_unet.py 
```

## Evaluation

For evaluation purposes an evaluation script using YoloV9 is provided. Given a directory of images, the script predicts the number of objects in each image in the directory and compares it to an expected count of objects.

To run the evaluation script, run the following:

1. Download yolov9e.pt (model weights):
```
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
```

2. Then, install evaluation requirements:
```
pip install -r evaluation_requirements.py
```

3. Finally, run the script.
```
python evaluation_script.py --images_dir examples --output evaluation_output
```

* `images_dir` : a path to a directory with images, in the format described below. For an example directory, see `examples/`.
* `output_dir` : a path to an output directory (automatically created if does not exist). The output of the script is a `results.csv` file with the predcitions, as well as annotated images with labeled bounding boxes.
 
In the directory with images, each image should have the following name format: `{expected_count}__{class_name}__{...}.png`, such as `4__donut__A_photo_of_four_donuts_on_the_road.png`.
* `{expected_count}` : an integer, representing the number of expected objects (e.g., 4). 
* `{class_name}` : a string, representing the class that is to be generated (e.g., donut).
* `{...}` : this is optional and can contain whatever, such as the prompt (e.g., A_photo_of_four_donuts_on_the_road).

