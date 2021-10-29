# Respiratory-Scans based COVID-19 Detection using Multi-Modal Multi-Task Learning Framework

## About

![MTLArch](https://user-images.githubusercontent.com/54277039/132307382-53335fdc-06ce-4c9b-b9d6-fa6a266cf219.png)

## Required Python Dependencies

These were categorized according to their application purposes as following :-

- Model Architectures and Deep Learning based Libraries
  - tensorflow
  - keras
- Evaluation Metrics and Statistical Analysis based Libraries
  - sklearn
  - scipy
- Image Processing and Data Visualization based Libraries
  - opencv (cv2)
  - pillow (PIL)
  - matplotlib
- Matrices and Data Engineering based Libraries
  - numpy
  - pandas
- System and Directory Management based Library
  - os
  - tqdm

## Adviced Structure of Dataset Folder

For the ease of Training and Implementation flow, following structure would be adviced :-

```
|--Dataset Folder/

   |--X_Ray_Train/

      |--COVID/
         
         |--x_c1_image_train
         |--x_c2_image_train
         ...
         
      |--Non_COVID/

         |--x_nc1_image_train
         |--x_nc2_image_train
         ...
         
   |--X_Ray_Test/

      |--COVID/
      
         |--x_c1_image_test
         |--x_c2_image_test
         ...
         
      |--Non_COVID/
      
         |--x_nc1_image_test
         |--x_nc2_image_test
         ...
         
   |--CT_Scan_Train/

      |--COVID/
      
         |--ct_c1_image_train
         |--ct_c2_image_train
         ...
         
      |--Non_COVID/
      
         |--ct_nc1_image_train
         |--ct_nc2_image_train
         ...

   |--CT_Scan_Test/

      |--COVID/
      
         |--ct_c1_image_test
         |--ct_c2_image_test
         ...
         
      |--Non_COVID/
      
         |--ct_nc1_image_test
         |--ct_nc2_image_test
         ...

```

## Steps for User-Customized Training

1. Arrange the Dataset according to the adviced structure, mentioned above.

2. Run Dataset_PreProcessing.py, which consist of 2 functions, naming Sample_Processing and Dataset_Processing, with the following details :-
   
   - Sample_Processing(x_ray_directory,ct_scan_directory) :
      - x_ray_directory --> A String of the X-Ray Image File Path with the name of the file and extension of it.
      - ct_scan_directory --> A String of the CT-Scan Image File Path with the name of the file and extension of it.
   
   - Dataset_Processing(x_base_directory,ct_base_directory) :
      - x_base_directory --> A String of the X-Ray Folder Path (till COVID or Non-COVID Folder).
      - ct_base_directory --> A String of the CT-Scan Folder Path (till COVID or Non-COVID Folder).

3. Run Embeddings_Generator.py, which consist of 2 functions, naming Embedding_Model and Embedding_Save, with the following details **(Replace Model_Name with keras-based Model Classes like keras.applications.VGG16, keras.applications.InceptionV3, etc.)** :-
   
   - Embedding_Model(Model_save_directory,Model_Class,Dataset_train_Directory,Dataset_test_Directory) :
      - Model_save_directory --> A String of the Transfer Learning Model Path, to be saved in future, with the name of file and extension (.h5 or .hdf5) of it.
      - Model_Class --> A Class Name of the Transfer Learning Model, like keras.applications.VGG16 (To be changed by the user in Function File)
      - Dataset_train_Directory --> A String of the Train Dataset Directory Path (till x_ray_train or ct_scan_train folder).
      - Dataset_test_Directory --> A String of the Train Dataset Directory Path (till x_ray_test or ct_scan_test folder).
   
   - Embedding_Save(Embed_save_Directory,Trained_Model_Class,Dataset_Directory,Image_Size) :
      - Embed_save_Directory --> A String of the Directory Path (with the name of file and extension (.npy) of it) where the Embeddings would be saved.
      - Trained_Model_Class --> The Class of the Trained Transfer Learning Model, which would be loaded from the user side using keras.models.load_model(<--Model_Path-->).
      - Image_Size --> The Size of images to be taken as input (331 for NasNetLarge and 224 for others) during embedding generation.
      - Dataset_Directory --> A String of the Dataset Directory Path (till x_ray_train, ct_scan_train, x_ray_test or ct_scan_test folder).

4. Run Task_Specific_Feature_Extractor.py, which consist of 2 functions, naming x_ray_task_specific and ct_scan_task_specific, with the following details :-
   
   - x_ray_task_specific(Model_save_directory,train_embed,test_embed,train_labels,test_labels) :
      - Model_save_directory --> A String of the X-Ray Task Specific Feature Extractor Path, to be saved in future, with the name of file and extension (.h5 or .hdf5) of it.
      - train_embed --> A numpy array of embeddings from training dataset images.
      - test_embed --> A numpy array of embeddings from testing dataset images.
      - train_labels --> A numpy array of task-specific labels (whether COVID or Non-COVID) from training dataset images.
      - test_labels --> A numpy array of task-specific labels (whether COVID or Non-COVID) from testing dataset images.
   
   - ct_scan_task_specific(Model_save_directory,train_embed,test_embed,train_labels,test_labels) :
      - Model_save_directory --> A String of the CT-Scan Task Specific Feature Extractor Path, to be saved in future, with the name of file and extension (.h5 or .hdf5) of it.
      - train_embed --> A numpy array of embeddings from training dataset images.
      - test_embed --> A numpy array of embeddings from testing dataset images.
      - train_labels --> A numpy array of task-specific labels (whether COVID or Non-COVID) from training dataset images.
      - test_labels --> A numpy array of task-specific labels (whether COVID or Non-COVID) from testing dataset images.

5. Decide whether to make Shared Features Extractors with or without Adversarial Training. Depending upon the choice, following would be the implementation :-

   - Without Adversarial Training, Run Shared_Features_Without_Adversarial.py, which consist of 2 functions, naming mlp_model and mlp_train, with the following details :-
      - mlp_model() --> It would make the MLP model with pre-defined parameters and architecture.
      - mlp_train(Model_save_Directory,x_train_embed,x_test_embed,ct_train_embed,ct_test_embed,train_labels,test_labels) --> A function to Train the pre-defined MLP Model with the help of training and testing embeddings of X-Ray and CT-Scan with their corresponding Task-Specific Labels (Whether COVID or Non-COVID), all being of the form of numpy array.

   - With Adversarial Training, Run Shared_Features_Extractor.py, which consist of some functions, with their following details :- 
      - gen_model() and disc_model() --> These Functions would produce the MLP Architecture with pre-defined parameters.
      - disc_loss_func(actual_embed,gen_embed,train_labels,invert_labels) --> Discriminator Loss Function where the actual input and generator's output go alongside Actual training labels and inverted labels (interchanging classes --> Whether X-Rays or CT-Scans), all of the form of numpy arrays.
      - gen_loss_func(gen_embed,invert_labels) --> Generator Loss Function where the Generated Output and inverted class labels (Whether X-Rays or CT-Scans) go, all of the form of numpy arrays.
      - train_step(actual_embed,old_gen_loss,old_disc_loss,train_labels,invert_labels) --> Adversarial Training happens here, alongside Actual Input, Old Values of Generator and Discriminator Loss, with Training Labels and Inverted Class Labels, all of the form of numpy arrays.
      - fine_tune(Model_save_Directory,model_gen,x_train_embed,x_test_embed,ct_train_embed,ct_test_embed) --> Here, the Shared Feature Extractor Module would be fine tuned and best model iteration would be saved according to "Model_save_Directory", with Class Labels (Whether X-Rays or CT-Scans).

6. Run Classifier_Head.py, which consist of 2 functions, naming Classifier_Head and Final_Embeddings, with the following details :-

   - Classifier_Head(Model_save_Directory,train_embed,test_embed,train_labels,test_labels) :
      - Model_save_Directory --> A String of the Classifier Head Path, to be saved in future, with the name of file and extension (.h5 or .hdf5) of it.
      - train_embed --> A numpy array of embeddings from training dataset images, after passing through Task Specific and Shared Features Extractor Modules.
      - test_embed --> A numpy array of embeddings from testing dataset images, after passing through Task Specific and Shared Features Extractor Modules.
      - train_labels --> A numpy array of Labels (Whether COVID or Non-COVID) for training dataset images.
      - test_labels --> A numpy array of Labels (Whether COVID or Non-COVID) for testing dataset images.
    
   - Final_Embeddings(<----Arguments---->) :
      - shared_model, x_task_model, ct_task_model --> Model Classes of Shared Features Extractor and Task-Specific X-Ray and CT-Scans Feature Extractors, loaded using keras.models.load_model(<--Model_Path-->).
      - shared_x_ray_train_input,shared_x_ray_test_input,shared_ct_train_input,shared_ct_test_input --> Numpy arrays of Embeddings Input for Shared Features Extractor Modules.
      - task_x_ray_train_input,task_x_ray_test_input,task_ct_train_input,task_ct_test_input --> Numpy arrays of Embeddings Input for Task-Specific Features Extractor Modules.

7. Run User_Customized_Pipeline.py, which consist of some functions, with the following details :-

   - Pre_Process(x,ct) :
      - x --> Greyscaled X-Ray Image Matrix, loaded using cv2.imread(<---Image Path---> , 0)
      - ct --> Greyscaled CT-Scan Image Matrix, loaded using cv2.imread(<---Image Path---> , 0)
   - Task_Specific_Embed(Save_Folder_Directory,x,ct) :
      - Save_Folder_Directory --> A String of the Container Folder Path where each component of the Pipeline was saved.
      - x --> Pre-Processed X-Ray Image Matrix.
      - ct --> Pre-Processed CT-Scan Image Matrix. 
   - Shared_Feature_Embed(Save_Folder_Directory,x,ct) :
      - Save_Folder_Directory --> A String of the Container Folder Path where each component of the Pipeline was saved.
      - x --> Pre-Processed X-Ray Image Matrix.
      - ct --> Pre-Processed CT-Scan Image Matrix.
   - Classification(Save_Folder_Directory,x,ct) : 
      - Save_Folder_Directory --> A String of the Container Folder Path where each component of the Pipeline was saved.
      - x --> Pre-Processed X-Ray Image Matrix.
      - ct --> Pre-Processed CT-Scan Image Matrix. 
   - Test_Run(Save_Folder_Directory,X_Ray_Grey_Image,CT_Scan_Grey_Image) : 
      - Save_Folder_Directory --> A String of the Container Folder Path where each component of the Pipeline was saved.
      - X_Ray_Grey_Image --> Greyscaled X-Ray Image Matrix, loaded using cv2.imread(<---Image Path---> , 0)
      - CT_Scan_Grey_Image --> Greyscaled CT-Scan Image Matrix, loaded using cv2.imread(<---Image Path---> , 0)

8. For Evaluating and Analysing the Generalization of Implementation, consider using Evaluation_and_Significance_Test.py, which consist of 2 functions, naming Evaluation_metrics and Significance_Test, with the following details :-
   
   - Evaluation_metrics(prob_list,labels,threshold=0.5) :
      - prob_list --> List of Output Probabilities after implementing the whole pipeline.
      - labels --> Output Classes vector.
      - threshold --> Confidence Probability (generally 0.5 for ideal sigmoid scenario).
   
   - Significance_Test(Save_Directory,task_train_embed,shared_train_embed,task_test_embed,shared_test_embed,train_labels,test_labels) :
      - Save_Directory --> A String of Sample Run Iterations, with the name and extension (.h5 or .hdf5) on it.
      - task_train_embed,shared_train_embed --> Numpy arrays of Task Specific and Shared Features Embeddings from Training Data.
      - task_test_embed,shared_test_embed --> Numpy arrays of Task Specific and Shared Features Embeddings from Testing Data.
      - train_labels,test_labels --> Numpy arrays of Training and Testing Labels (Whether COVID or Non-COVID)

## Pre-Processed Train-Test Dataset

In order to start with the suggested Dataset from the Research Paper, access this link and use it accordingly, with Label Annotations as train.csv and test.csv. As it was already pre-processed, Running Dataset_PreProcessing.py won't be a requirement.

https://drive.google.com/drive/folders/11YKYq6Go6RmIXziIYts5nB97e9LVstzd?usp=sharing

## Test Run using Pre-Trained Pipeline

Run Pre_Trained_Pipeline.py with the Directory Paths mentioned accordingly.

Pre-Trained Model Weights ---> https://drive.google.com/drive/folders/15-Mtxa5xtQzt-5WVYO17u9uixWlcXO8l?usp=sharing

## Documentations and Resources

1. https://keras.io/api/applications/
2. https://docs.opencv.org/4.5.1/d2/d96/tutorial_py_table_of_contents_imgproc.html
3. https://www.youtube.com/watch?v=tX-6CMNnT64&list=LL&index=102
4. https://docs.scipy.org/doc/scipy/reference/stats.html
5. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
6. https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
