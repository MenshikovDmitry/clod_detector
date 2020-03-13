import cv2,os
import numpy as np


def brightness_and_contrast(image, brightness=50, contrast=40):
    '''
    adjusts brightness and contrast of the image
    '''
    img = np.int16(image)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def my_preprocessor(image):
    '''
    crops the image to square and adjusts brigtness
    '''
    min_dim=min(image.shape[:2])
    frame_image=image[:min_dim,:min_dim,:]
    return brightness_and_contrast(frame_image, 60,40)


def frames_from_video(filename, out_path, number_of_images, 
                      frame_start=0, frame_end=0, image_preprocessor=None):
    '''
    generates <number_of_images> images from video file. 
    choses them evenly from the entire video, 
    bounded by <frame_start> and <frame_end>
    image_processor: function for preprocessing images before saving

    '''
    out_filename=os.path.basename(filename).replace('.','_')

    cap=cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print ("could not open :",video_filename)
        return None

    video_len=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_end==0: frame_end=video_len

    frames_step=(video_len-frame_start-(video_len-frame_end))//number_of_images

    frame_number=0
    ret=True
    while True and ret:
        ret,frame=cap.read()
        if ret and frame_number>=frame_start and frame_number<frame_end:
            frame_number+=1
            if frame_number%frames_step==0:
                #saving the frame
                full_out_path=os.path.join(out_path,out_filename+'_'+str(frame_number)+'.jpg')
                if not image_preprocessor==None:
                    frame=image_preprocessor(frame)
                cv2.imwrite(full_out_path,frame)
                print('.',end="")
    cap.release()   
    print('Done')


if __name__ == '__main__':
    filename=os.path.join('data','fonar.avi')
    out_path=os.path.join('data','images')
    
    frames_from_video(filename, out_path,
                        50,
                        frame_end=0,
                        image_preprocessor=my_preprocessor)