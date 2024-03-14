import keras
import cv2
import numpy as np
from LicencePlateMask.src.keras_utils import load_model
from LicencePlateMask.src.utils import im2single
from LicencePlateMask.src.keras_utils import load_model, detect_lp
from LicencePlateMask.src.label import Shape

class LincencePlateMask():
    """ licence_plate_mask_model_path: Yolo trained weights model path
        image: CV2 image in numpy array form
        plate_mask_logo : template image of mask logo to be replaced on licence plate
        output: returns numpy array image with licence plate masked """
    
    def __init__(self,image_mask_logo,licence_plate_mask_model_path):

        self.wpod_net = load_model(licence_plate_mask_model_path)
        self.mask_logo = image_mask_logo

    def adjust_pts(self,pts,lroi):
        return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


    def NumPlateMask(self,image,lp_threshold = 0.9):
        """ color : if template image is not found then color to be filled on licence plate""" 
        # height and width of image
        h= image.shape[0]
        w= image.shape[1]
        h1,w1 = self.mask_logo.shape[:2]
    
        pts0 = np.array([[0,0],
                        [w1,0],
                        [w1,h1],
                        [0,h1]])
        # input required for yolo model 
        ratio = float(max(image.shape[:2]))/min(image.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)

        # Model detection
        Llp,LlpImgs,_ = detect_lp(self.wpod_net,im2single(image),bound_dim,2**4,(240,80),lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)
            for shape in [s]:
                ptsarray = shape.pts.flatten()

                #Making points for making polyfill in cv2
                pts = np.array([[[int(ptsarray[0]*w),int(ptsarray[4]*h)]],
                                [[int(ptsarray[1]*w),int(ptsarray[5]*h)]],
                                [[int(ptsarray[2]*w),int(ptsarray[6]*h)]],
                                [[int(ptsarray[3]*w),int(ptsarray[7]*h)]]])
                # try:
                # Finding prospect using Homography to replace licence plate with logo.
                hgt, mask = cv2.findHomography(pts0, pts, cv2.RANSAC,5.0)

                # adding logo at plate area
                im1Reg = cv2.warpPerspective(self.mask_logo, hgt, (w,h))

                mask2 = np.zeros(image.shape, dtype=np.uint8)

                roi_corners2 = np.int32([[int(ptsarray[0]*w),int(ptsarray[4]*h)],
                                        [int(ptsarray[1]*w),int(ptsarray[5]*h)],
                                        [int(ptsarray[2]*w),int(ptsarray[6]*h)],
                                        [int(ptsarray[3]*w),int(ptsarray[7]*h)]])

                channel_count2 = image.shape[2]  
                ignore_mask_color2 = (255,)*channel_count2

                # filling zeros at plate area place on white image mask2
                mask2 = cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
                mask2 = cv2.bitwise_not(mask2)

                # adding zeros on plate area
                masked_image2 = cv2.bitwise_and(image, mask2)

                #Using Bitwise or to merge the two images at lincence plate area
                image= cv2.bitwise_or(im1Reg, masked_image2)

                return image
        else:
            return image
                