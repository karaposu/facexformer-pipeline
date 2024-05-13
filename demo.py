
from image_input_handler import  UniversalImageInputHandler
# from facexformer_pipeline import facexformer_pipeline

from facexformer_pipeline.facexformer_pipeline import facexformer_pipeline

def main():

    image_path = "sample_image.jpg"
    uih = UniversalImageInputHandler(image_path, debug=False)
    COMPATIBLE, img = uih.COMPATIBLE, uih.img

    facexformer_pipeline(img)

if __name__ == "__main__":

    main()