Programming Language:
Python 3.12

Code Structure:
The code consists of a decision tree implementation for binary classification based on the specified option: 'optimized', 'randomized', 'forest3', or 'forest15'. The main components include a TreeNode class representing nodes in the decision tree, functions for reading data, calculating entropy and information gain, and methods for tree training and testing.

How to Run the Code:
1. Ensure you have Python Version 3.12 installed.
2. Open a terminal and navigate to the directory containing the code.
3. Run the program using the following command:
   python dtree.py training_file test_file option

Example:
python dtree.py pendigits_training.txt pendigits_test.txt optimized

Running on ACS Omega:
Note: The code does not require compilation and should run on ACS Omega without issues.

Additional Notes:
The code includes options for decision tree training and testing based on the specified criteria.
Pruning is performed during tree construction if a validation set is provided.
Results are saved in "output.txt" with details for each test object and overall classification accuracy.

