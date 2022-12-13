# Transfer-Learning
The aim of this project is to implement a flower classifier using transfer learning. <br>
In this program, we used `MobileNetV2` as our base model. <br>
The dataset we used includes 1000 images with 5 classes. <br>
Two versions to implement transfer learning: **standard version** and **accelerated version**.

### Standard transfer learning
  1.Replace the last layer of `MobileNetV2` with a dense layer. 
  2.Freeze all the layer without the last layer
*task3 ~ task8* in the code.

### Accelerated transfer learning
  1.Prepare a precomputed dataset by using `MobileNetV2` model without last layer.
  2.Build a model with only 1 input layer and 1 dense layer.
*task9 ~ task10* in the code.
