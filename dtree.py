import numpy as np
from collections import Counter
import sys

class TreeNode:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def predict(self, example):
        if self.label is not None:
            return self.label
        else:
            if example[self.attribute] <= self.threshold:
                return self.left.predict(example)
            else:
                return self.right.predict(example)

def read_data(file_path):
    data = np.genfromtxt(file_path, filling_values=0)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Class labels
    return X, y

def entropy(y):
    class_counts = Counter(tuple(map(tuple, y.tolist())))
    total_instances = len(y)
    entropy_value = 0.0

    for count in class_counts.values():
        probability = count / total_instances
        entropy_value -= probability * np.log2(probability + 1e-10)  # Add a small value to avoid log(0)

    return entropy_value

def information_gain(examples, attribute, threshold):
    total_instances = len(examples)
    left_mask = examples[:, attribute] <= threshold
    right_mask = ~left_mask

    left_entropy = entropy(examples[left_mask])
    right_entropy = entropy(examples[right_mask])

    left_weight = np.sum(left_mask) / total_instances
    right_weight = np.sum(right_mask) / total_instances

    information_gain_value = entropy(examples) - (left_weight * left_entropy + right_weight * right_entropy)
    return information_gain_value

def choose_attribute_optimized(examples, attributes):
    max_gain = best_attribute = best_threshold = -1

    for attribute in attributes:
        attribute_values = examples[:, attribute]
        L = np.min(attribute_values)
        M = np.max(attribute_values)

        for k in range(1, 51):
            threshold = L + k * (M - L) / 51
            gain = information_gain(examples, attribute, threshold)

            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                best_threshold = threshold

    return best_attribute, best_threshold

def choose_attribute_randomized(examples, attributes):
    random_attribute = np.random.choice(attributes)
    threshold = np.median(examples[:, random_attribute])
    return random_attribute, threshold

def select_attribute(examples, attributes, option):
    if option == 'optimized':
        return choose_attribute_optimized(examples, attributes)
    elif option == 'randomized':
        return choose_attribute_randomized(examples, attributes)
    else:
        raise ValueError("Invalid option for tree training")

def build_tree(examples, attributes, option, depth=0, max_depth=10, validation_data=None):
    if depth >= max_depth or len(set(examples[:, -1])) == 1 or not examples.any() or examples.shape[1] < 2:
        # Create a leaf node
        if not examples.any():
            print(f"Empty examples array at depth {depth}")
            label = Counter(examples[:, -1]).most_common(1)[0][0]
            return TreeNode(label=label)
        elif examples.shape[1] < 2:
            print(f"Invalid examples array at depth {depth}")
            label = Counter(examples[:, -1]).most_common(1)[0][0]
            return TreeNode(label=label)

        label = Counter(examples[:, -1]).most_common(1)[0][0]
        return TreeNode(label=label)

    attribute, threshold = select_attribute(examples, attributes, option)
    print(f"Depth: {depth}, Selected Attribute: {attribute}, Threshold: {threshold}")

    left_mask = examples[:, attribute] <= threshold
    right_mask = ~left_mask

    if np.all(left_mask) or np.all(right_mask):
        # All examples belong to one side, create a leaf node
        label = Counter(examples[:, -1]).most_common(1)[0][0]
        return TreeNode(label=label)

    left_child = build_tree(examples[left_mask], attributes, option, depth + 1, max_depth, validation_data)
    right_child = build_tree(examples[right_mask], attributes, option, depth + 1, max_depth, validation_data)

    # Pruning: Check if pruning is beneficial
    if validation_data is not None:
        # Calculate accuracy before pruning
        accuracy_before_pruning = test_decision_tree(left_child, validation_data)

        # Prune the subtree and calculate accuracy after pruning
        left_child_accuracy = test_decision_tree(left_child, validation_data)
        right_child_accuracy = test_decision_tree(right_child, validation_data)

        if left_child_accuracy > right_child_accuracy:
            print(f"Pruning at depth {depth} - Choosing left child.")
            return left_child
        else:
            print(f"Pruning at depth {depth} - Choosing right child.")
            return right_child

    return TreeNode(attribute=attribute, threshold=threshold, left=left_child, right=right_child)

def train_decision_tree(examples, attributes, option, max_depth=10, validation_data=None):
    # Implementation of decision tree training based on the specified option
    if option == 'optimized':
        return build_tree(examples, attributes, option, max_depth=max_depth, validation_data=validation_data)
    elif option == 'randomized':
        return build_tree(examples, attributes, option, max_depth=max_depth, validation_data=validation_data)
    elif option in ['forest3', 'forest15']:
        num_trees = 3 if option == 'forest3' else 15
        forest = []
        for _ in range(num_trees):
            tree = build_tree(examples, attributes, 'randomized', max_depth=max_depth, validation_data=validation_data)
            forest.append(tree)
        return forest
    else:
        raise ValueError("Invalid option for tree training")

def test_decision_tree(tree, test_data, overwrite=False):
    mode = 'w' if overwrite else 'a'
    
    correct_predictions = 0
    total_instances = len(test_data)
    accuracies = []

    with open("output.txt", mode) as output_file:
        for i, example in enumerate(test_data):
            predicted_class = tree.predict(example)
            true_class = example[-1]

            if isinstance(tree, TreeNode):
                # For decision tree
                accuracy = 1 if predicted_class == true_class else 0
            else:
                # For decision forest
                accuracy = 1 if true_class in predicted_class else 0

            accuracies.append(accuracy)

            output_file.write(f"Object Index = {i}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}\n")

            correct_predictions += accuracy

        classification_accuracy = sum(accuracies) / total_instances
        output_file.write(f"\nClassification Accuracy = {classification_accuracy}\n")

    # print(f"Results {'overwritten' if overwrite else 'appended'} to output.txt.")

    return classification_accuracy

def main():
    if len(sys.argv) != 4:
        print("Usage: dtree training_file test_file option")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    option = sys.argv[3]

    # Read data
    training_data = read_data(training_file)
    test_data = read_data(test_file)

    # Get attributes (excluding the class label)
    attributes = list(range(training_data[0].shape[1] - 1))

    # Split training data into training and validation sets
    split_index = int(0.8 * len(training_data[0]))
    training_set, validation_set = training_data[0][:split_index], training_data[0][split_index:]

    # Train decision tree
    if option in ['optimized', 'randomized']:
        decision_tree = train_decision_tree(training_set, attributes, option, validation_data=validation_set)
        test_decision_tree(decision_tree, test_data[0])
    elif option in ['forest3', 'forest15']:
        forest = train_decision_tree(training_set, attributes, option, validation_data=validation_set)
        for i, tree in enumerate(forest):
            print(f"\nTree {i + 1}:")
            test_decision_tree(tree, test_data[0])
    else:
        print("Invalid option. Please choose 'optimized', 'randomized', 'forest3', or 'forest15'.")

if __name__ == "__main__":
    main()