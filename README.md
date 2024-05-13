# FaceXFormer Pipeline Implementation

This repository contains the Python implementation of the FaceXFormer, a unified transformer model for comprehensive facial analysis, as described in the paper by Kartik Narayan et al. from Johns Hopkins University. This implementation focuses on providing a modular, easy-to-use interface for the model to perform various facial analysis tasks.

## Features

- **Unified Architecture**: Integrates multiple facial analysis tasks into a single transformer-based model.
- **Flexible Task Handling**: Allows dynamic configuration of tasks such as face parsing, landmark detection, head pose estimation, and more.
- **Robustness**: Effectively handles images "in-the-wild," proving strong generalizability and robustness across diverse conditions.
- **High Performance**: Maintains real-time performance with the capability to process images at 37 FPS.

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.6+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt



#Usage

To use the FaceXFormer pipeline, follow these steps:

Import the pipeline class:
from facexformer_pipeline import FacexformerPipeline

Initialize the pipeline with desired tasks:
pipeline = FacexformerPipeline(debug=True, tasks=['headpose', 'landmark', 'attributes'])


Run the model on an image:
results = pipeline.run_model(image_array)

Access the results:
print(results['headpose'])
print(results['landmark_list'])


#Acknowledgements

This implementation is based on the research done by Kartik Narayan and his team at Johns Hopkins University. All credit for the conceptual model and its validation belongs to them.
