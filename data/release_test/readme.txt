There are 3352 testing images. For each image, a prediction mask of scene parsing should be generated, in which each pixel has its predicted class index (class idx ranges from 1 to 150). Also the size of some prediction mask should be exactly the same as its testing image.

Please save the prediction mask as the file name of the tesing image with png format. For example, for the testing image ADE_test_00003489.jpg, the name of its prediction mask should be ADE_test_00003489.png.

All the prediction masks should be compressed into a zip file, then submitted to ILSVRC evaluation server before the due. Please make sure each png file could be read by both matlab and python correctly, otherwise the unreadable prediction mask will be ignored then the final score will be affected.

Contact Bolei Zhou (bzhou@csail.mit.edu) if you have any questions.
Sep.4, 2016
