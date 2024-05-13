import os
import sys
script_dir = os.path.dirname(__file__)  # Directory of the current script
sys.path.append(script_dir)  # Appe
sys.path.append(os.path.join(os.path.dirname(__file__), 'facexformer'))

from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN
from image_input_handler import  UniversalImageInputHandler
from utils import denorm_points, unnormalize, adjust_bbox, visualize_head_pose, visualize_landmarks, visualize_mask
from task_postprocesser import task_faceparsing,  process_landmarks, task_headpose , task_attributes, task_gender, process_visibility
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import numpy as np
import torch

class FacexformerPipeline:
    TASK_MAP = {
        'faceparsing': 0,
        'landmark': 1,
        'headpose': 2,
        'attributes': 3,
        'age_gender_race': 4,
        'visibility': 5
    }
    def __init__(self, model_path=None, debug=False, tasks=None):
        self.debug=debug
        self.labels=self.initialize_labels()
        if model_path is None:
            if self.check_if_file_exists("ckpts/model.pt"):
                model_path = "./ckpts/model.pt"
                if self.debug:
                    print("Model weights found locally")
            else:
                self.download_trained_model()
                model_path = "./ckpts/model.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.debug:
            print("Device set to:", self.device)
        self.model = self.load_model(model_path)
        self.active_tasks = self.map_tasks(tasks)

    def map_tasks(self, tasks):
        if tasks is None:
            return list(range(6))  # Default to all tasks
        return [self.TASK_MAP[task] for task in tasks if task in self.TASK_MAP]

    def check_if_file_exists(self, file_path):
        return os.path.exists(file_path)
        # add logic here

    def download_trained_model(self):
        try:
            hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
        except Exception as e:
            if self.debug:
                print("Failed to download model:", e)
            raise

    def load_model(self, weights_path):
        model = FaceXFormer().to(self.device)
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict_backbone'])
            if self.debug:
                print("Model loaded successfully.")
        except Exception as e:
            print("Failed to load model:", e)
            return None
        model.eval()
        return model

    def transform_image(self, image):
        transformations = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transformations(image)

    def crop_face_area_from_image(self, image):
        mtcnn = MTCNN(keep_all=True)
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            x_min, y_min, x_max, y_max = boxes[0]
            width, height = image.size
            x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)
            return image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        if self.debug:
            print("No faces detected.")
        return image  # Return unmodified image if no face is detected

    def initialize_labels(self):
        label_shapes = {
            "segmentation": (224, 224),
            "lnm_seg": (5, 2),
            "landmark": (68, 2),
            "headpose": (3,),
            "attribute": (40,),
            "a_g_e": (3,),
            "visibility": (29,)
        }
        return {key: torch.zeros(shape) for key, shape in label_shapes.items()}

    def prepare_for_model(self, image, labels):
        image = image.unsqueeze(0).to(self.device)
        labels = {k: v.unsqueeze(0).to(self.device) for k, v in labels.items()}
        return image, labels

    def process_task_output(self, task_id, output, results, images):
        if task_id == 0:
            results['faceparsing_mask'] = task_faceparsing(output[7])
        elif task_id == 1:
            results['landmark_list'] = process_landmarks(output[0], images)
        elif task_id == 2:
            results['headpose'] = task_headpose(output[1], images)
        elif task_id == 3:
            results['attributes'] = task_attributes(output[2])
        elif task_id == 4:
            results['age_gender_race_dict'] = task_gender(output[4], output[5], output[6])
        elif task_id == 5:
            results['visibility'] = process_visibility(output[3])
    def run_model(self, image, image_is_cropped=True):
        results = {}

        image = Image.fromarray(image)
        if not image_is_cropped:
            image = self.crop_face_area_from_image(image)
        image = self.transform_image(image)
        model_ready_image, labels = self.prepare_for_model(image, self.labels)

        for i in self.active_tasks:
            task = torch.tensor([i]).to(self.device)
            output = self.model(model_ready_image, labels, task)
            self.process_task_output(i, output, results, model_ready_image)

        image = unnormalize(model_ready_image[0].detach().cpu())
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = image[:, :, ::-1]
        results['image'] = image
        return results

def main():
    image_path = "sample_image.jpg"
    uih = UniversalImageInputHandler(image_path, debug=False)
    COMPATIBLE, img = uih.COMPATIBLE, uih.img
    pipeline = FacexformerPipeline(debug=False, tasks=['headpose', 'landmark', 'attributes'])
    results = pipeline.run_model(img)

    print(results["headpose"])

    # results=pipeline.run_model(uih.img)
    # print(results["headpose"])
    # print(results["age_gender_race_dict"])
    # print(results["visibility_result"])

if __name__ == "__main__":

    main()
