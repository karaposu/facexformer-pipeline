
from image_input_handler import  UniversalImageInputHandler
# from facexformer_pipeline import facexformer_pipeline
from facexformer_pipeline.facexformer_pipeline import FacexformerPipeline
from visual_debugger import VisualDebugger, Annotation, AnnotationType
import numpy as np

def main():
    vd = VisualDebugger(tag="facexformer", debug_folder_path="./", active=True)
    image_path = "sample_image.jpg"
    uih = UniversalImageInputHandler(image_path, debug=False)
    COMPATIBLE, img = uih.COMPATIBLE, uih.img
    print('COMPATIBLE:', COMPATIBLE)

    pipeline = FacexformerPipeline(tasks=['landmark', 'headpose'], debug=True)
    results = pipeline.run_model(uih.img)

    transformed_image = results['transformed_image'].numpy()
    transformed_image = np.transpose(transformed_image, (1, 2, 0))
    unnormalized_image = results['image']

    # print(results['headpose'])

    landmarks_annotation = [Annotation(type=AnnotationType.POINTS,  coordinates=results["landmark_list"], color=(0, 255, 0))]
    scaled_landmarks_annotation = [ Annotation(type=AnnotationType.POINTS, coordinates=results["scaled_landmarks"], color=(0, 255, 0))]

    vd.visual_debug(img, landmarks_annotation, process_step="landmarks" )
    vd.visual_debug(img, scaled_landmarks_annotation, process_step="scaled_landmarks")

    vd.visual_debug(transformed_image, landmarks_annotation, process_step="landmarks_on_transformed")
    vd.visual_debug(unnormalized_image, landmarks_annotation, process_step="landmarks_on_unnormalized_image ")

    vd.visual_debug(transformed_image, scaled_landmarks_annotation, process_step="landmarks_on_transformed")
    vd.visual_debug(unnormalized_image, scaled_landmarks_annotation, process_step="landmarks_on_unnormalized_image ")


# annotations = [Annotation(type=AnnotationType.PITCH_YAW_ROLL,
    #                           coordinates=None,
    #                           orientation=(
    #                           results["headpose"]["pitch"], results["headpose"]["yaw"], results["headpose"]["roll"]),
    #                           color=(0, 255, 0))]
    # vd.visual_debug(img, annotations, process_step="head_orientation")

if __name__ == "__main__":

    main()