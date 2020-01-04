# Links

https://en.wikipedia.org/wiki/Collaborative_filtering  
https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0  
https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)  
https://towardsdatascience.com/paper-summary-matrix-factorization-techniques-for-recommender-systems-82d1a7ace74  
https://www.youtube.com/watch?v=ZspR5PZemcs  
http://nicolas-hug.com/blog/matrix_facto_1  


# Content

**mf<nolink>.py** - Class used to implement the algorithm. It takes a matrix as input and computes (_train_ function) an approximation for all its values (_get_rating_ & _get_full_matrix_ functions), based on the non-zeros values the matrix contains.  
**splitSubsets<nolink>.py** - Splits the MovieLens dataset into a training and a testing subset.  
**train<nolink>.py** - Trains the model using mf<nolink>.py and saves the output.  
**test<nolink>.py** - Uses the trained model from the above script to test it's accuracy.  
**main<nolink>.py** - Downloads the dataset from MovieLens url and runs the above scripts in order.  
