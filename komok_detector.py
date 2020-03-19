import os,cv2
from matplotlib import pyplot as plt

import tensorflow as tf

import custom
import mrcnn.model as modellib
from mrcnn import visualize

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Override the training configurations with a few
# changes for inferencing.
config = custom.CustomConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Komok_Detector():
    
    def __init__(self,model_weights,device='/gpu:0'):
        
        self.config=InferenceConfig()
        self.device=device
        self.model_dir=os.path.join(os.getcwd(),'logs')
        
        with tf.device(self.device):
            self.model = modellib.MaskRCNN(mode="inference", 
                                           model_dir=self.model_dir,
                                           config=self.config)
        self.model.load_weights(model_weights, by_name=True)
        self.colors=[(1,0,0) for i in range (20)]
    
    
    def predict(self,image):
        results = self.model.detect([image], verbose=0)
        return results
    
    def visualize(self,image,predictions):
        r = predictions[0]
        #ax = get_ax(1)
        fig= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['komok','komok'], r['scores'], 
                     #       ax=ax,
                            title="Predictions",
                            colors=self.colors,
                            show_mask=False,
                            show_bbox=True,
                            show_polygon=True,
                            mask_biggest=True)
        return fig
        
        
    def detect (self,image):
        predictions=self.predict(image)
        return self.visualize(image,predictions)
            


if __name__=="__main__":
    import argparse
    import cv2
    import numpy as np
    import mimetypes

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/komok_weights.h5",
                        help="Path to weights .h5 file.")
    parser.add_argument("input",
                        metavar="image.jpeg",
                        help="input file for detection. Image or video")
    parser.add_argument("--output",
                        metavar="/path/to/output",
                        help="folder for saving the output images")
    args = parser.parse_args()

    weights= args.weights
    filename=args.input
    output=args.output

    print("Weights: ",weights)
    print("input file: ",filename)
    print("output to: ",output)

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', 800,800)
    detector=Komok_Detector(weights)
    
    #checking the type of input file
    if 'image' in mimetypes.guess_type(filename)[0]:
        #the type of input is image
        image=cv2.imread(args.input)
        print("Processing single image. Press ESC to close the window")

        img=detector.detect(image)
        #show the result in a pop up window
        while True:
            cv2.imshow('main', img)
            if cv2.waitKey(1) == 27:
                break

    elif 'video' in mimetypes.guess_type(filename)[0]:
        #the type of input is videofile
        print("Processing video. Press ESC to close the window")

        #opening the videofile
        cap=cv2.VideoCapture(filename)
        if not cap.isOpened(): 
            print ("could not open :",filename)
            exit()
        ret=True
        i=0
        while ret:
            i+=1
            ret,frame=cap.read()
            if ret:
                img=detector.detect(frame)
                cv2.imshow('main', img)
                if output:
                    out_filename=os.path.basename(filename).replace('.','_')+'_'+str(i)+'.jpg'
                    cv2.imwrite(os.path.join(output,out_filename),img)
                
                if cv2.waitKey(1) == 27:
                    break
        
    cv2.destroyAllWindows()