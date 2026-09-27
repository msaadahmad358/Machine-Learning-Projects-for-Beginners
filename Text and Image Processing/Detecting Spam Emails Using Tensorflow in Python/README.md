Email spam detection is a machine learning project using tensorflow and complete it in multiple steps.

Step 1 - Import required libraries
Step 2 - Load the dataset
Step 3 - Balance the dataset
Step 4 - Clean the text
Step 5 - Tokenization and padding
Step 6 - Define the model
Step 7 - Train the model

EarlyStopping and ReduceLROnPlateau is used to let the model stop early if performance does'nt improve and reduce the learning rate to fine-tune the model.

Dataset has Unnamed, label, text and label_num. (5171, 4)
dataset link: https://media.geeksforgeeks.org/wp-content/uploads/20250320162008521713/spam_ham_dataset.csv

Some results are presented below:
Test loss:  0.1284964680671692
Test accuracy:  0.9700000286102295

This was the whole process, followed in this project to build the machine learning model using TensorFlow to detect whether email is spam or non-spam. Results are saced in the outputs using plt.savefig in jpg form.