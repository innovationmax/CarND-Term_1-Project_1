# **Finding Lane Lines on the Road** 

## Project Description

The purpose of this project is to detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.


[//]: # (Image References)

[image1]: ./\Pipeline_steps/step1.jpg "Step 1, Image is converted to grayscale"
[image2]: ./\Pipeline_steps/step2.jpg "Step 2, a Gaussian Blur filter is applied to smooth out edges"
[image3]: ./\Pipeline_steps/step3.jpg "Step 3, Canny algorithm resulting edges detection"
[image4]: ./\Pipeline_steps/step4.jpg "Step 4, focusing on a region of iterest part of the image"
[image5]: ./\Pipeline_steps/step5.jpg "Step 5, applying Hough transform to image"
[image6]: ./\Pipeline_steps/step6.jpg "Step 6, consolidated left and right lane lines"
[image7]: ./\Pipeline_steps/step7.jpg "Step 7, resulting image with lanes marked with red lines"


---

### Reflection

### 1. Pipeline Description.
To accomplish the task of lane detection in images, the following 7 steps are performed in order:

1. Image file is loaded and is converted to gray-scale as shown below, this steps is necessary as the canny algorithm we will requires input in gray-scale format.

![image1]

2. Then next step is to apply a Gaussian blurring filter to smooth out the picture. this step is optional and could be omitted, since the canny edge algorithm infernally applies 5*5 filter, the resulting picture after the filter is shown below.

![image2]

3. In this step the Canny algorithm is applied to the image to detect edges. Canny algorithm works by calculating the gradient and finding points with the highest gradient value. an example result from this step is shown below.

![image3]

4. Since the camera position is in a fixed position on the vehicle, we could improve the pipeline performance by including edge detection results from a particular region in the image, in my pipeline I use, an Isosceles trapezoid shape, with the top and bottom base lengths calculated using a top skew ratio and bottom skew ratio applied to the image width, the hight is determined using a hight scale ratio applied to the total hight of the image. the following code snippet shows outlines the application of this concept. and the example image shows the resulting image after applying this region of interest calculation method.

```python
top_skew_ratio = 0.04
bottom_skew_ratio = 1
height_ratio = 0.6
height = imshape[0]
width = imshape[1]
vertices = np.array([[(width*(1 - bottom_skew_ratio)/2, height),
                      (width*(1 - top_skew_ratio)/2, height_ratio*height),
                      (width*(1 + top_skew_ratio)/2, height_ratio*height),
                      (width*(1 + bottom_skew_ratio)/2, height)]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)
```

![image4]

5. The result from the edge detection is individual pixels, applying Hough transform to this result would look in the image and group the pixels that belong to the same line, and result in line segments presented in the form of line objects with start and end (x1,y1),(x2,y2) points. the algorithm is calibrated with threshold to find lines with a particular minimum number of pixels and also looks at consolidating line segments if the distance between them is below a particular gap value. results from this application is shown in image below.

![image5]

5. With the output from Hough transform, a draw lines function is developed to categorize the line segments between left and right lane and drawing only two lines for lane detection. this categorization is done based on the slope of the individual line segments make sure to include lines past a certain slope value to filter out any horizontal and extreme slope segments. Polynomial fit is applied to the two groups of starting and ending points, but also takes into account the length of the line segments and uses it as a weight in the fitting function, this application improves the accuracy of the detected lane marks by making sure it follows the strongest line segments detected. the draw line function is shown below a long with an example image result.

```python
def draw_lines(img, lines, top_y, bottom_y, color=[255, 0, 0], thickness=4):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
        
    line segments will be grouped in left lane and right lane, 
    and polyfit will be used to get the approximate line, line 
    length will be used as weights for the fitting function,
    this would insure that the fit would take into account how many
    points were used from the image to create the line in the first place 
    """
    
    left_lane_x = []
    left_lane_y = []
    left_lane_weight = []
    
    right_lane_x = []
    right_lane_y = []
    right_lane_weight = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # calculate line segmant slope and length
            slope = (y2 - y1) / (x2 - x1)
            length = (((y2 - y1)**2) + ((x2 - x1)**2))**(0.5)
            # ignore line with small slope, this would get rid of horizontal lines 
            if math.fabs(slope) < 0.5:
                continue
            # negative slope indicates line belongs to left lane
            if slope <= 0:
                left_lane_x.extend([x1, x2])
                left_lane_y.extend([y1, y2])
                left_lane_weight.extend([length, length])
            # positive slope indicates line belongs to right lane
            else:
                right_lane_x.extend([x1, x2])
                right_lane_y.extend([y1, y2])
                right_lane_weight.extend([length, length])
   
    
    # polynomial fit is used to get the line equation for left and right lanes
    # the function is made as x = f(y), this will allow of the calculation of x
    # given the top_y and bottom_y values passed to this function
    y1 = int(bottom_y)
    y2 = int(top_y)
    
    if left_lane_y and left_lane_x:
        left_lane = np.poly1d(poly.polyfit(left_lane_y, left_lane_x, deg=1, w=left_lane_weight)[::-1])
        x1_left = int(left_lane(y1))
        x2_left = int(left_lane(y2))
        cv2.line(img, (x1_left, y1), (x2_left, y2), color, thickness)
    if right_lane_y and right_lane_x:
        right_lane = np.poly1d(poly.polyfit(right_lane_y, right_lane_x, deg=1, w=right_lane_weight)[::-1])  
        x1_right = int(right_lane(y1))
        x2_right = int(right_lane(y2))
        cv2.line(img, (x1_right, y1), (x2_right, y2), color, thickness)
            
```
![image6]

7. Finally the detected lines are overlayed on top of the original image and is output from the pipeline function as shown in the example image below.

![image7]

### 2. Identify potential shortcomings with your current pipeline

Pipeline was applied to the two video streams included in the assignment, and it performed very well in both of them, however you notice that when the vehicle is coming up to a bend in the road, the bend curvature is not detected since the pipeline is designed to look only for straight line lane markings. for a self driving application this would have to be modified to detected such curvatures and use their data to control vehicle steering. 

For the challenge portion of the assignment, it is noticed that for particular section of the road the road colors were different and would require different calibration for the pipeline to work effectively. 

### 3. Suggest possible improvements to your pipeline

The pipeline would be improved by accommodating a broader range of road color and lighting conditions, and by including additional detection functions to handle road bends and curves. 