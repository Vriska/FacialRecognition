# Facial Recognition 

Uses pre trained embedding system by Google (FaceNet) to extract facial embeddings which are then
Collected into a pickle file and classifier are trained, gridsearch enables rapid testing of various 
Classification pramaters to come to what would be the best classifier. The creation of the embeddings 
Via image files has been done separately than training the classifier because the same set of embeddings can
Be mappes to multiple classifiers the names and mapped vectors have been dumped by joblib sperately and
A classifiers trainer has been done sperately. 
