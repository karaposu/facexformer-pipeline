import os
import sys
script_dir = os.path.dirname(__file__)  # Directory of the current script
sys.path.append(script_dir)  # Appe
sys.path.append(os.path.join(os.path.dirname(__file__), 'facexformer'))


# import sys
# import facexformer_pipeline.facexformer_pipeline
#
# # Delete the module from sys.modules
# del sys.modules['facexformer_pipeline.facexformer_pipeline']

from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN
from image_input_handler import  UniversalImageInputHandler
from facexformer_pipeline.utils import denorm_points, unnormalize, adjust_bbox, visualize_head_pose, visualize_landmarks, visualize_mask
from task_postprocesser import task_faceparsing,  process_landmarks, task_headpose , task_attributes, task_gender, process_visibility
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from visual_debugger import VisualDebugger, Annotation, AnnotationType
import cv2

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

        self.mtcnn = MTCNN(keep_all=True)

        self.vdebugger = VisualDebugger(tag="facex", debug_folder_path="./", active=True)



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


    def find_head_ROI_coordinates(self,
                               face_coordinates,
                               chin_extension_ratio,
                               top_margin_ratio,
                               left_margin_ratio,
                               right_margin_ratio):

        chinflexoffset = chin_extension_ratio
        topflexoffset = top_margin_ratio
        leftflexoffset = left_margin_ratio
        rightflexoffset = right_margin_ratio

        x, y, w, h = face_coordinates
        # try:
        #     x, y, w, h = face_coordinates
        # except Exception as e:
        #     print("Error unpacking fd_coordinates:", e)
        #     raise

        # print("y: ", y)
        # print("chinflexoffset: ", chinflexoffset)
        # print("h: ", h)
        bottom_cut_line_coordinate = y + int(h * (1 + chinflexoffset))
        # Apply right and left flex offsets
        right_cut_line = x + int(w * (1 + rightflexoffset))
        left_cut_line = x - int(w * leftflexoffset)
        # Correct for potential out-of-bound values on the top
        if (y - h * topflexoffset) < 0:
            top_cut_line_coordinate = 0
        else:
            top_cut_line_coordinate = int(y - h * topflexoffset)
        # Correct for potential out-of-bound values on the left
        if left_cut_line < 0:
            left_cut_line = 0

        # Calculate new height and width of the ROI
        new_h = bottom_cut_line_coordinate - top_cut_line_coordinate
        new_w = right_cut_line - left_cut_line

        # Defining new coordinates for the crop
        coordinates = (left_cut_line, top_cut_line_coordinate, new_w, new_h)


        return coordinates

    def crop_rect_ROI_from_Img(self, img, coordinates):
        x, y, w, h = coordinates
        rect = img[(y): (y + h), x: x + w]

        return rect

    def calculate_head_ROI(self,image, fd_coordinates):

        bottom_margin_ratio = 0.30
        top_margin_ratio = 0.30
        left_margin_ratio = 0.30
        right_margin_ratio = 0.30

       # print(">>>>>>>>>>>>>fd_coordinates:",fd_coordinates )

        head_ROI_coordinates=self.find_head_ROI_coordinates( fd_coordinates,
                                                    bottom_margin_ratio,
                                                    top_margin_ratio,
                                                    left_margin_ratio,
                                                    right_margin_ratio)

        head_ROI = self.crop_rect_ROI_from_Img(image, head_ROI_coordinates)

        return  head_ROI, head_ROI_coordinates

        # msg = "head_ROI shape : {}".format(head_ROI.shape)
        # logger.debug(msg)
        # msg = "head_ROI coordinates : {}".format(str(head_ROI_coordinates))
        # logger.debug(msg)

        # self.head_ROI = head_ROI
        # self.head_ROI_coordinates = head_ROI_coordinates




    def calculate_face_ROI(self):
        head_ROI = self.crop_rect_ROI_from_Img( self.img, head_ROI_coordinates)


        # face_ROI = self.crop_head_ROI_from_Img(self.img, self.fd_coordinates)
        #
        #
        # self.face_ROI=face_ROI

    def transform_image(self, image):
        transformations = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transformations(image)

    def fd(self, image, one_point_format=False):
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            x_min, y_min, x_max, y_max = boxes[0]
            #fd=[int(x_min), int(y_min), int(x_max), int(y_max)]

            if one_point_format:
                x = int( x_min)
                y = int(y_min)
                w = int(x_max - x_min)
                h = int(y_max - y_min)
                fd= [ x, y, w, h]

            return fd

    def obtain_various_crops_of_the_image(self,image, fd_coordinates):

        face_ROI = self.calculate_face_ROI(image, fd_coordinates)
        head_ROI, head_ROI_coordinates = self.calculate_head_ROI(image, fd_coordinates)




    # def crop_face_area_from_image(self, image):
    #     # mtcnn = MTCNN(keep_all=True)
    #     boxes, _ = self.mtcnn.detect(image)
    #     if boxes is not None:
    #         x_min, y_min, x_max, y_max = boxes[0]
    #         width, height = image.size
    #         x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)
    #         return image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    #     if self.debug:
    #         print("No faces detected.")
    #     return image  # Return unmodified image if no face is detected

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

    def place_mask_in_original_image(self,original_image, head_mask, face_coords):
        # Get original image dimensions
        orig_height, orig_width = original_image.shape[:2]

        # Create an empty mask with the same size as the original image
        full_size_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

        # Get the face coordinates
        x, y, w, h = face_coords

        # Place the head mask into the full-size mask
        full_size_mask[y:y + h, x:x + w] = head_mask

        return full_size_mask


    def convert_local_to_global(self, local_landmarks, roi_start_point):
        # roi_start_point now is expected to be (left_cut_line, top_cut_line_coordinate)
        global_landmarks = []
        for lm in local_landmarks:
            global_x = lm[0] + roi_start_point[0]
            global_y = lm[1] + roi_start_point[1]
            global_landmarks.append((global_x, global_y))
        return global_landmarks

    # def process_task_output(self, original_image, task_id, output, results):
    def process_task_output(self, original_image, face_ROI, face_coordinates, head_ROI, head_ROI_coordinates, task_id, output, already_cropped,  results):
        results["face_coordinates"]= face_coordinates
        results["head_ROI"] = head_ROI
        results["face_ROI"] = face_ROI
        results["head_coordinates"] = head_ROI_coordinates


        if task_id == 0:
            normalized_faceparsing_mask = task_faceparsing(output[7])
            results['normalized_faceparsing_mask']=normalized_faceparsing_mask

            white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
            annotations = [Annotation(type=AnnotationType.MASK, mask=normalized_faceparsing_mask, color=(0, 255, 0))]
            # self.vdebugger.visual_debug(white_image, annotations, name="pure_faceparsing_mask")

            # results['normalized_faceparsing_mask']

            head_ROI_size = head_ROI.shape[:2]
            resized_mask = cv2.resize(normalized_faceparsing_mask, (head_ROI_size[1], head_ROI_size[0]), interpolation=cv2.INTER_NEAREST)

            results['faceparsing_mask_head_ROI']=resized_mask
            full_size_mask = self.place_mask_in_original_image(original_image, resized_mask, head_ROI_coordinates)
            results['faceparsing_mask'] = full_size_mask


            # unique_values, frequencies = np.unique(resized_mask, return_counts=True)
            # for value, count in zip(unique_values, frequencies):
            #     print(f"mask valuesss: {value}, Frequency: {count}")





        elif task_id == 1:
            results['landmark_list'] = process_landmarks(output[0])
            # x=denorm_points(results['landmark_list'], 224, 224, align_corners=False)

            #results['landmarks'] =self.scale_landmarks_to_original_image(original_image,results['landmark_list'] )
            results['landmarks_face_ROI'] =self.scale_landmarks_to_original_image(face_ROI,results['landmark_list'] )
            if already_cropped:
                results['landmarks'] = results['landmarks_face_ROI']
            else:
                results['landmarks']= self.convert_local_to_global( results['landmarks_face_ROI'], face_coordinates)

            # results['landmarks'] = self.scale_landmarks_to_original_image(original_image, x)
        elif task_id == 2:
            results['headpose'] = task_headpose(output[1])
        elif task_id == 3:
            results['attributes'] = task_attributes(output[2])
        elif task_id == 4:
            results['age_gender_race_dict'] = task_gender(output[4], output[5], output[6])
        elif task_id == 5:
            results['visibility'] = process_visibility(output[3])

    # from PIL import Image, ImageDraw
    # import numpy as np

    def scale_landmarks_to_original_image(self, original_image, landmarks, resized_image_size=(224, 224)):
        # print("landmarks[0]:", landmarks[0])
        original_width, original_height = original_image.shape[1],  original_image.shape[0]
        # print("original_width:", original_width, "original_height:", original_height)
        resized_width, resized_height = resized_image_size
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        # Scale landmarks back to original image dimensions


        scaled_landmarks = [(x * scale_x, y * scale_y) for (x, y) in landmarks]
        scaled_landmarks_int = [(int(round(x)), int(round(y))) for (x, y) in scaled_landmarks]

        # print("scaled_landmarks[0]:", scaled_landmarks[0])

        return scaled_landmarks_int

    def run_model(self, image, face_coordinates=None, already_cropped=False,  head_coordinates=None ):


        # already cropped means, already face cropped

        #   sc1 : only landmarks, big image,   (f d= None, already_cropped=False)
        #?? sc2 : only landmarks, already cropped image.  run_model(image, already_cropped=True)
        #   sc3 : landmarks, headpose  big image  (f d= None, already_cropped=False)
        #   sc4 : landmarks,  big image  (f d= [10,10,10,10], already_cropped=False)

        #   sc5 : landmarks, headpose  big image  (f d= [10,10,10,10], already_cropped=False)


        results = {}
        head_ROI= None
        original_image = image.copy()

        if face_coordinates is None:
            if not already_cropped:
                face_coordinates=self.fd(image, one_point_format=True)
                print("-----------fd_coordinates:",face_coordinates )

                annotations = [ Annotation(type=AnnotationType.RECTANGLE, coordinates=face_coordinates, color=(0, 255, 0))]
                # self.vdebugger.visual_debug(original_image, annotations, name="fd_on_org_image")


                face_ROI = self.crop_rect_ROI_from_Img(image, face_coordinates)

                # self.vdebugger.visual_debug(face_ROI, name="face_ROI")


                transformed_face_ROI = self.transform_image(Image.fromarray(face_ROI))
                # import cv2
                debug_transformed_face_ROI = cv2.resize(face_ROI, (224, 224), interpolation=cv2.INTER_CUBIC)

                # self.vdebugger.visual_debug(debug_transformed_face_ROI, name="transformed_face_ROI")

                model_ready_face_ROI_image, labels = self.prepare_for_model(transformed_face_ROI, self.labels)
                if 0 in self.active_tasks:
                    head_ROI, head_ROI_coordinates = self.calculate_head_ROI(image, face_coordinates)
                    transformed_head_ROI = self.transform_image(Image.fromarray(head_ROI))
                    model_ready_head_ROI_image, labels = self.prepare_for_model(transformed_head_ROI, self.labels)

            else:
                face_ROI= image
                head_ROI = image
                if 0 in self.active_tasks:
                    transformed_head_ROI = self.transform_image(Image.fromarray(head_ROI))
                transformed_face_ROI = self.transform_image(Image.fromarray(face_ROI))
                model_ready_face_ROI_image, labels = self.prepare_for_model(transformed_face_ROI, self.labels)

        else:
            face_ROI = self.crop_rect_ROI_from_Img(image, face_coordinates)
            transformed_face_ROI = self.transform_image(Image.fromarray(face_ROI))
            model_ready_face_ROI_image, labels = self.prepare_for_model(transformed_face_ROI, self.labels)
            if 0 in self.active_tasks:
                head_ROI, head_ROI_coordinates = self.calculate_head_ROI(image, face_coordinates)
                transformed_head_ROI = self.transform_image(Image.fromarray(head_ROI))
                model_ready_head_ROI_image, labels = self.prepare_for_model(transformed_head_ROI, self.labels)

        # if head_ROI is not None:
        #     model_ready_head_ROI_image, labels = self.prepare_for_model(transformed_head_ROI, self.labels)

        for i in self.active_tasks:
            task = torch.tensor([i]).to(self.device)
            if i in [0] :
                output = self.model(model_ready_head_ROI_image, labels, task)
            else:
                output = self.model(model_ready_face_ROI_image, labels, task)
            # self.process_task_output(original_image, i, output, results)
            self.process_task_output(original_image, face_ROI, face_coordinates,  head_ROI, head_ROI_coordinates,  i, output,already_cropped,  results)



        if not already_cropped:
            pass

            # annotations = [Annotation(type=AnnotationType.POINTS, coordinates=results["landmark_list"], color=(0, 255, 0))]
            # self.vdebugger.visual_debug(debug_transformed_face_ROI,annotations,  name="raw_lm_on_transformed")
            #
            # annotations = [Annotation(type=AnnotationType.POINTS, coordinates=results["landmarks_face_ROI"], color=(0, 255, 0))]
            # self.vdebugger.visual_debug(face_ROI, annotations, name="scaled_lm_on_face_ROI")

        # annotations = [Annotation(type=AnnotationType.POINTS, coordinates=results["landmarks"], color=(0, 255, 0))]
        # self.vdebugger.visual_debug(original_image, annotations, name="scaled_lm_on_org_img")

       # face_ROI=model_ready_face_ROI_image[0].detach().cpu()

        # unnormalized_model_ready_face_ROI_image = unnormalize(model_ready_face_ROI_image[0].detach().cpu())
        # unnormalized_model_ready_head_ROI_image = unnormalize(model_ready_head_ROI_image[0].detach().cpu())
        # results['transformed_face_ROI'] = model_ready_face_ROI_image[0].detach().cpu()
        # results['transformed_head_ROI'] = model_ready_head_ROI_image[0].detach().cpu()
        # results['unnormalized_face_ROI'] = unnormalized_model_ready_face_ROI_image
        # results['unnormalized_head_ROI'] = unnormalized_model_ready_head_ROI_image

        return results

        # image = Image.fromarray(image)
        # if not image_is_cropped:
        #     image = self.crop_face_area_from_image(image)
        # image = self.transform_image(image)
        # model_ready_image, labels = self.prepare_for_model(image, self.labels)

    # def run_model(self, image, image_is_cropped=True):
    #     results = {}
    #     original_image=image.copy()
    #     image = Image.fromarray(image)
    #     if not image_is_cropped:
    #         image = self.crop_face_area_from_image(image)
    #     image = self.transform_image(image)
    #     model_ready_image, labels = self.prepare_for_model(image, self.labels)
    #
    #     for i in self.active_tasks:
    #         task = torch.tensor([i]).to(self.device)
    #         output = self.model(model_ready_image, labels, task)
    #         self.process_task_output(i, output, results, model_ready_image)
    #
    #     image = unnormalize(model_ready_image[0].detach().cpu())
    #     image = image.permute(1, 2, 0).numpy()
    #     image = (image * 255).astype(np.uint8)
    #     image = image[:, :, ::-1]
    #     results['image'] = image
    #     results['transformed_image'] = model_ready_image[0]
    #     if 1 in self.active_tasks:
    #         results['landmarks'] =self.scale_landmarks_to_original_image(original_image,results['landmark_list'] )
    #     return results

def main():

    image_path1 = "sample_image_head_only.jpg"

    uih = UniversalImageInputHandler(image_path1, debug=False)
    img1 = uih.img

    pipeline = FacexformerPipeline(debug=True, tasks=['headpose', 'landmark', 'faceparsing'])

    results = pipeline.run_model(img1)

    vdebugger = VisualDebugger(tag="facex", debug_folder_path="./", active=True)

    annotation_landmarks_face_ROI = [
        Annotation(type=AnnotationType.POINTS, coordinates=results["landmarks_face_ROI"], color=(0, 255, 0))]
    annotation_landmarks = [Annotation(type=AnnotationType.POINTS, coordinates=results["landmarks"], color=(0, 255, 0))]
    annotation_headpose = [
        Annotation(type=AnnotationType.PITCH_YAW_ROLL, orientation= [results["headpose"]["pitch"], results["headpose"]["yaw"], results["headpose"]["roll"]], color=(0, 255, 0))]
    annotation_face_coordinates = [
        Annotation(type=AnnotationType.RECTANGLE, coordinates=results["face_coordinates"], color=(0, 255, 0))]
    annotation_head_coordinates = [
        Annotation(type=AnnotationType.RECTANGLE, coordinates=results["head_coordinates"], color=(0, 255, 0))]
    annotation_faceparsing = [Annotation(type=AnnotationType.MASK, mask=results["faceparsing_mask"], color=(0, 255, 0))]
    annotation_faceparsing_head_ROI = [ Annotation(type=AnnotationType.MASK, mask=results["faceparsing_mask_head_ROI"], color=(0, 255, 0))]

    vdebugger.visual_debug(img1, name="original_image")
    vdebugger.visual_debug(img1, annotation_face_coordinates, name="", stage_name="face_coor")
    vdebugger.visual_debug(results["face_ROI"], name="", stage_name="cropped_face_ROI")
    vdebugger.visual_debug(img1, annotation_head_coordinates, name="", stage_name="head_coor")
    vdebugger.visual_debug(results["head_ROI"], name="", stage_name="cropped_head_ROI")
    vdebugger.visual_debug(results["face_ROI"], annotation_landmarks_face_ROI, name="landmarks", stage_name= "on_face_ROI")
    vdebugger.visual_debug(img1, annotation_landmarks, name="landmarks", stage_name= "on_image")
    vdebugger.visual_debug(results["face_ROI"], annotation_headpose, name="headpose")
    vdebugger.visual_debug(results["head_ROI"], annotation_faceparsing_head_ROI,name="faceparsing",  stage_name="mask_on_head_ROI")
    vdebugger.visual_debug(img1, annotation_faceparsing, name="faceparsing", stage_name="mask_on_full_image")
    vdebugger.cook_merged_img()







if __name__ == "__main__":

    main()
