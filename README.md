# AI_Final_project
The repository that stores the code for my project
in the course Artificial Intelligence.<br><br>
The code should be run through runfile.py,
using command line arguments. The arguments may specify what data it should run on
(either the full or missing data), which model to run,
whether to use "better features" (which means fewer features due to
anomaly detection and PCA)
and whether to run the parameter maximizing functions
(only available for random forest and neural network).<br><br>
The model argument should be lower case and spaces should be swapped with underscores
(like spectral_clustering) except for DBSCAN and KMeans.
There is also an option to enter "odor_based" as the model to get a plot
of the data according to the odor (the tags).
Also, the findParams argument should only get "true" or "false", and the data
argument should be either "whole" or "missing".<br><br>
When creating clustering visualizations, they will be saved with the model name 
succeeded by "visualization", to differentiate between the visualizations
that came with the repository, and the ones that were just created.<br><br>
If you'd like to run part of the models and function with different parameters,
you may run the other files directly with whatever parameters
or whatever functions you'd like, as the code itself has a lot more option than can be 
accessed through the runfile alone.
