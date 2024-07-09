# FaceXFormer Pipeline Implementation

This repository contains the easy-to-use pipeline implementation of the FaceXFormer, a unified transformer model for comprehensive facial analysis, as described in the paper by Kartik Narayan et al. from Johns Hopkins University. 

Here is official code repo : https://github.com/Kartik-3004/facexformer

![Alt text]("https://github.com/karaposu/facexformer-pipeline/blob/main/0_merged.png?raw=true")

What this implementation does differently?

Official implementation is awesome as it is but mainly focuses on benchmarking and therefore it is not application ready yet. With this implementation: 

- No need to deal with reverse trasnforms or resizing or remapping to original image size
- cropping is handled for you (different crops used for faceparsing and landmarks for better accuracy )
- possible to run one task or any combination of tasks. 
- you can pass your own face detection method's coordinates as argument and you are not forced to rerun the face detection
- visual debugging is a lot easy thanks to use visual_debugger package
- results are ready with all extra information you may need


## What is it 

You can use FaceXFormer to extract
- **faceparsing mask**
- **landmarks**
- **headpose orientation**
- **various attributes** 
- **visibility** 
- **age-gender-race** 

information really fast and from unified model.  And you can do it really fast (37 FPS).


## Installation
   ```bash
   pip install facexformer_pipeline 
   ```

## Usage

To use the FaceXFormer pipeline, follow these steps:

```bash
#Import the pipeline class:

from facexformer_pipeline import FacexformerPipeline

#Initialize the pipeline with desired tasks:
pipeline = FacexformerPipeline(debug=True, tasks=['headpose', 'landmark', 'faceparsing'])


# here is your code for reading an image 
# image_array

#Run the model on an image:
results = pipeline.run_model(image_array)

#Access the results:
print(results['headpose'])
print(results['landmarks'])


#Show the results on image :
from visual_debugger import VisualDebugger, Annotation, AnnotationType
vdebugger = VisualDebugger(tag="facex", debug_folder_path="./", active=True)

annotation_landmarks_face_ROI= [Annotation(type=AnnotationType.POINTS, coordinates= results["landmarks_face_ROI"], color=(0, 255, 0))]
annotation_landmarks= [Annotation(type=AnnotationType.POINTS, coordinates= results["landmarks"], color=(0, 255, 0))]
annotation_headpose = [Annotation(type=AnnotationType.PITCH_YAW_ROLL, orientation= results["headpose"], color=(0, 255, 0))]
annotation_face_coordinates = [Annotation(type=AnnotationType.RECTANGLE, coordinates= results["face_coordinates"], color=(0, 255, 0))]
annotation_head_coordinates = [Annotation(type=AnnotationType.RECTANGLE, coordinates= results["head_coordinates"], color=(0, 255, 0))]
annotation_faceparsing = [Annotation(type=AnnotationType.MASK, mask= results["faceparsing_mask"], color=(0, 255, 0))]
annotation_faceparsing_head_ROI = [Annotation(type=AnnotationType.MASK, mask= results["faceparsing_mask_head_ROI"], color=(0, 255, 0))]


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




```

# Acknowledgements

This implementation is based on the research done by Kartik Narayan and his team at Johns Hopkins University. All credit for the conceptual model and its validation belongs to them.
