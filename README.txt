Edith Flores
A Simple VSM Search Engine

Files:
searchEngine.py
readme.txt

--------searchEngine.py-----------------
The program preprocess.py takes an image file and a number. Seg takes each pixel in the image file as a data sample and takes k as the number of means, and send them to the kmeans program to determine the k-mean clusters, and which their corresponding data samples. Seg creates a new image with the corresponding k mean clusters. The output is an image window with the original image and the new k-mean clustered image, the new image is then saved as imgK.gif.
To run:
python3 searchEngine.py

The rest of the document files are needed to preprocess, some have been altered to implement documents as queries.
searchEngine.py is the program.
