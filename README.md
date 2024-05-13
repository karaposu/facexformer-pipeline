# FaceXFormer Pipeline Implementation

This repository contains the easy-to-use pipeline implementation of the FaceXFormer, a unified transformer model for comprehensive facial analysis, as described in the paper by Kartik Narayan et al. from Johns Hopkins University. 

Here is official code repo : https://github.com/Kartik-3004/facexformer

## What is it 

You can use FaceXFormer to extract
- **landmarks**
- **headpose orientation**
- **various attributes** 
- **visibility** 
- **age-gender-race** 
information really fast and from unified model.  And you can do it really fast(37 FPS).


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
pipeline = FacexformerPipeline(debug=True, tasks=['headpose', 'landmark', 'attributes'])


#Run the model on an image:
results = pipeline.run_model(image_array)

#Access the results:
print(results['headpose'])
print(results['landmark_list'])
```

# Acknowledgements

This implementation is based on the research done by Kartik Narayan and his team at Johns Hopkins University. All credit for the conceptual model and its validation belongs to them.
