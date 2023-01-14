# LandmarkClassificationForSocialMedia
Using pythorch to create  2 solutions to the Landmark Classification problem in Udacity Deep learning course. 

The first solution has a hand defined model in PyTorch, the requirements for passing this project was 50%+ accuracy on this model.
It has 7 Convolutional Layers and
       4 MaxPooling layers
       BatchNorm after each layer
       ReLU activation function
       
All this feeds 64 * 14* 14 features after being flattened to the head, which also includes a dropout. This achieved an accuracy of 58%,
which is an optimal result for the lenght of training it had and the complexity of the dataset we used.

The second solution was using transfer learning the resnet18 model supplied directly by pytorch and achieved a percentage accuracy of 74%
