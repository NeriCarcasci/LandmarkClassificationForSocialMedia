# Landmark Classification For Social Media
Using pythorch to create  2 solutions to the Landmark Classification problem in Udacity Deep learning course. 

The first solution has a hand defined model in PyTorch, the requirements for passing this project was 50%+ accuracy on this model.
It has 7 Convolutional Layers and
       4 MaxPooling layers
       BatchNorm after each layer
       ReLU activation function
       
All this feeds 64 * 14* 14 features after being flattened to the head, which also includes a dropout. This achieved an accuracy of 58%,
which is an optimal result for the lenght of training it had and the complexity of the dataset we used.

The second solution was using transfer learning the resnet18 model supplied directly by pytorch and achieved a percentage accuracy of 74%


## Project Specifications

In this project, you will apply the skills you have acquired in the Convolutional Neural Network (CNN) course to build a landmark classifier.

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.

In this project, you will take the first steps towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. You will go through the machine learning design process end-to-end: performing data preprocessing, designing and training CNNs, comparing the accuracy of different CNNs, and deploying an app based on the best CNN you trained.
