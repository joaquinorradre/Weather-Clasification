#set par(justify: true)
#set text(
  size: 16pt
)
#align(alignment.center)[= Weather Classification]
#set text(
  size: 12pt
)
== Dataset Analysis
is a *Weather Image Classification* dataset sourced from #link("https://www.kaggle.com/datasets/jehanbhathena/weather-dataset")[
  Kaggle dataset
]
It contains images categorized into different *weather conditions*, aimed at training and evaluating classification models.  
This dataset contains labeled *6862 images* of different types of weather separated in *eleven classes*. All images were taken with professional cameras, leading to similar image sizes and quality. We will take a deeper look into that aspect in the following part.
=== Examples from each class
#figure(
  image("reports/figures/dataset_analysis/example_from_each_class.svg"),
  caption: [Example from each class]
)

=== Class distribution in each set
//#image("reports/figures/dataset_analysis/class_distribution_barchart.svg", width: 80%)
The distribution across the classes is not uniform. We can take a look at the imbalance by analysing this stacked barchart.
#figure(
  image("reports/figures/dataset_analysis/class_distribution_stacked.svg", width:60%),
  caption: [Class distribution in a stacked barchart]
)
The first thing that is noticeable is that the class rime is the most populated and the rainbow class is the least represented, followed by lightning. The rest, do not have that much variability. This is a factor that will be taken into account when doing the cross entropy loss in our training, using the option to feed it weights for each class depending on their representation on the dataset.

On this plot we can also see that the whole dataset has been divided into 3 sets: train, validation and test. The train will be used to train all the models, taking an 80% of the original dataset. Then, the validation set, taking a 10% of the original set, will be used to see how each model generalises and choose the best model and hyperparameters. Finally, the test dataset will be used at the very end to see the generalization of the best model. When making the different subsets, the original distribution of the classes was respected.

=== Image sizes
Looking at this scatter plot of image dimensions across the dataset, there is a clear concentration of smaller images clustered in the lower-left region (roughly 200-1200 pixels), but with notable outliers extending to much larger dimensions, some reaching 5000x3000 pixels. 
The extreme range means some images will undergo heavy downsampling while others require minimal resizing, which could introduce inconsistencies in how the model learns features.
#figure(
  image("reports/figures/dataset_analysis/image_size_overall.svg", width:60%),
  caption: [Image size distribution]
)
Some weather phenomena seem to have characteristic image sizes. For instance, larger images appear more frequently for certain classes like "frost," "fogsmog," and "rime," while the dense lower-left cluster contains mixed classes. This could indicate these larger images come from different sources or capture contexts. It is worth noticing that very large images (3000x3000+) might contain fine details important for distinguishing similar weather conditions (e.g., frost vs. rime). Aggressive downsampling could hurt performance.
=== Mean image per class
Another analysis we can perform is the mean image per class. This might not seem very useful since each image has different objects and backgrounds. However, we can make some hypotheses.
#figure(
  image("reports/figures/dataset_analysis/mean_image_per_class.svg", width:60%),
  caption: [Mean image per class]
)
What stands out the most is the colour of the "dew", "sandstorm" and "lightning" classes. If you take a look at the examples from each class, we can assume that these colours are prominent because on the dew class has zoomed plants pictures, that explains the green colour. On the lightning, in order to capture them in a picture, the exposure is low, that explains the darkness of the pictures. Finally, the sandstorm has that orange tint because of the sand colour. The rest of the classes have either a gray tint because of the rain clouds or a blueish colour because of the sky.

== Problem Approach


== Performance Analysis
=== Metrics (Loss and Accuracy)
=== Architectures
==== MLP
==== LeNetModern
==== CNN V1
==== CNN V2
==== CNN V3