# What’s in a face? Facial appearance associated with emergence but not success in entrepreneurship

# Abstract
Facial appearance has been associated with leader selection in domains where effective leadership is considered crucial, such as politics, business and the military. Few studies, however, have so far explored associations between facial appearance and entrepreneurship, despite the growing expectation that societies project on entrepreneurs for providing exemplary leadership in activities leading to the creation of disruptive start-ups. By using computer vision tools and a large-scale sample of entrepreneurs and non-entrepreneurs from Crunchbase, we investigate whether three geometrically based facial characteristics - facial width-to-height ratio (fWHR), cheekbone prominence, and facial symmetry - as well as advanced statistical models of whole facial appearance, are associated with a) the likelihood of an individual to emerge as an entrepreneur and b) the performance of the company founded by that individual. We find that cheekbone prominence, facial symmetry and two whole facial appearance statistical models are associated with the likelihood of an individual to emerge as an entrepreneur. In contrast to entrepreneurship emergence, none of the examined facial characteristics are associated with performance. Overall, our results suggest that facial appearance is associated with the emergence of leaders in the entrepreneurial endeavor, however, it is not informative about their subsequent performance.

# fWHR, cheekbone prominence and facial symmetry
We use Face++ to measure the facial width-to-height ratio (fWHR), the cheekbone prominence, and two measures of facial symmetry: overall facial asymmetry and central facial asymmetry. To derive these facial measurements, we used 16 facial landmarks/points (see Fig. a). Points P1 (left eye left corner) and P3 (left eye right corner) are the corners of the left eye while P2 (right eye right corner) and P4 (right eye left corner) are the corners of the right eye. Points P5 (face contour left) and P6 (face contour right) are the widest central points of a face contour. The two points are on the horizontal line below the eyes. P7 and P8 are the left and right point of the nose in the lowest nose part. P11 and P12 are the left corner and the right corner of the mouth respectively. P9 and P10 are defined as the width of the face (jaw width) at the same horizontal line as the points P11 and P12. Point P13 is the chin and P14 is the top point of the upper lip of the mouth. The final points P15 (left eyebrow right corner) and P16 (right eyebrow left corner) are the innermost eyebrow corners.

![1-s2 0-S1048984321001028-gr2_lrg](https://github.com/dstefa02/Facial-Structure/assets/8780840/0b8eebd6-f124-4e36-947a-e09c37606228)

**Calculation of fWHR:** We calculated fWHR using the bizygomatic breadth (width) divided by the distance between the middle of the eyebrows to the center of the upper lip (height) (Hodges-Simeon et al., 2016, Carré and McCormick, 2008). In Fig. b, “a” represents the width and “b” the height and the division “a/b” represents the fWHR. For the calculation of width we used the points P5 and P6 in Fig. a. For the height, we first calculated the midpoint of the P15 and P16 and then calculate its distance to P14. In robustness tests, we also examined a different operationalization of the fWHR, called the fWHR-lower, which uses the bizygomatic breadth (width) divided by the distance between the mean eye height and the bottom of the chin (height of the lower face) (Hodges-Simeon et al., 2016, Lefevre et al., 2012). The results were qualitatively similar.

**Calculation of cheekbone prominence:** We calculated cheekbone prominence using the bizygomatic breadth divided by the jaw width (width at the corners of the mouth) (Hodges-Simeon et al., 2016). In Fig. c, “a” represents the upper width and “b” the lower width and the division “a/b” represents the cheekbone prominence. For the upper width we used the points P5 and P6 and for the lower width we used the points P9 and P10 in Fig. a.

**Calculation of facial asymmetry:** In our study, we focused on horizontal asymmetry using the overall facial asymmetry (OFA) and the central facial asymmetry (CFA) metrics (Grammer & Thornhill, 1994). Both of these metrics are based on the sum of the differences between pairs of midpoints, where on a perfectly symmetrical face all midpoints must be on the same vertical line. We adapted the formulas of OFA and CFA in such a way that our results will not be affected by the resolution of the photos or by the distance of the faces from the camera (please see Appendix A). Thus, instead of computing the “differences between midpoints”, we computed the “divisions between midpoints”. We used the following formula (1), to calculate the OFA, where D1, D2, D3, D4, D5 and D6 refer to the lines in Fig. d. According to this formula, a perfectly symmetrical face has an OFA value equal to 0.(1)
![image](https://github.com/dstefa02/Facial-Structure/assets/8780840/7b1a2ba4-abab-48ad-9d79-63e8938c36ba)

In order to calculate CFA, we used the following formula, where D1, D2, D3, D4, D5 and D6 refer to the lines in Fig. d and where a perfectly symmetrical face has a CFA value equal to 0.(2)
![image](https://github.com/dstefa02/Facial-Structure/assets/8780840/784dbe85-ed4b-4bab-9fdc-bcf99e339096)

# Facial recognition using the whole face
![1-s2 0-S1048984321001028-gr3_lrg](https://github.com/dstefa02/Facial-Structure/assets/8780840/7fdb39b2-b9b5-4b26-a95b-d9535628f386)

Drawing from this line of work (Zhao et al., 2003, Stoker et al., 2016), we examine whether the whole facial appearance is related to the emergence of entrepreneurs and their company performance. By taking a whole-face model into account, we are taking advantage of a high-dimensional space that captures generic properties of faces like darkness/lightness and face shape (Blanz and Vetter, 2003, Kemelmacher-Shlizerman and Basri, 2010). To investigate the relationship between whole facial appearance and entrepreneurship emergence and firm performance, we further preprocessed the images of our dataset using Face++ and OpenCV (https://opencv.org/) (Bradski & Kaehler, 2008) (an open-source, computer vision and machine learning software library).

For the first approach, each image was cropped, resized) and transformed to a vector of 2000 elements, where each element corresponds to the cartesian coordinates of each facial landmark identified by Face++ on that image. For the second approach, each image was cropped, resized, converted to grayscale, processed to remove background noise, had its intensity normalized (histogram equalization), and transformed to a vector of 18,000 elements, where each element corresponds to the intensity (between 0 and 255) of each pixel of the image.

Next, we used k-fold cross validation (Mosteller and Tukey, 1968, Stone, 1974, Geisser, 1993, Kohavi, 1995) in order to split our dataset to training and test sets. k-fold cross validation is a standard resampling procedure, which is widely used to a) evaluate machine learning models on a limited data sample without overfitting and b) stack/combine various machine learning models into one more potent model (Wolpert, 1992). k-fold cross validation technique avoids overfitting, reducing the likelihood that a split will result in sets that are not representative of the full data set (Lever, Krzywinski, & Altman, 2016). The technique randomly splits the data at hand into k groups/sets (k = 10) from which 1 group is the test set while the remaining 9 groups are the training set. This procedure is repeated 10-times so that all groups have been selected once and only once as a test set. In each of the 10 iterations, first we train a two-stage dimensionality reduction algorithm (PCA and then LDA1) using the 9 sets of training data and then we transform/reduce the test set. Finally, all the transformed/reduced test sets are combined into an output vector, which is then used as an independent variable in our models.

In particular, the above procedure is repeated for each of the two whole facial modeling approaches, and each output vector is used as a different independent variable - “Holistic-based approach (whole-pixels)” and “Feature-based approach (facial landmarks)” respectively.

# Dependencies
Crunchbase_Collector.py:
  - pymongo
  - grequests
  - requests
  - pillow
    
Image_Preprocessing_and_Facial_Measures.py:
- numpy
- opencv-python
- shapely
- scipy
- matplotlib
- scikit-learn

# Citation
Stefanidis, D., Nicolaou, N., Charitonos, S. P., Pallis, G., & Dikaiakos, M. (2022). What’s in a face? Facial appearance associated with emergence but not success in entrepreneurship. The Leadership Quarterly, 33(2), 101597.

# License
All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See LICENSE.md for the full license text.
