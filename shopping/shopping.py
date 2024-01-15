import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # we open the csv file as a reader and naviguate trough it :) 
    #

    label_list = []
    evidence_list = []
    with open(filename,'r') as csv_file : 
        csv_reader = csv.reader(csv_file)

        next(csv_reader)
        # declare the return list : 
        for line in csv_reader :   
            # append the elements to the list :
            evidence_list.append(line[:-1])
            if line[-1] == 'TRUE':
                label_list.append(1)
                continue
            else:
                label_list.append(0)
                continue

    for element in evidence_list :

        for i in range(17):

            if  i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                element[i] = float(element[i])
                continue
            # match the mounths to their corresponding integer

            elif i == 10:
                if element[10] == 'Jan':
                    element[10] = 0
                    continue
                elif element[10] =='Feb':
                    element[10] = 1
                    continue
                elif element[10] == 'Mar':
                    element[10] = 2
                    continue
                elif element[10] == 'Apr':
                    element[10] = 3
                    continue
                elif element[10] == 'May':
                    element[10] = 4
                    continue
                elif element[10] == 'June':
                    element[10] = 5
                    continue
                elif element[10] == 'Jul':
                    element[10] = 6
                    continue
                elif element[10] == 'Aug':
                    element[10] = 7
                    continue
                elif element[10] == 'Sep':
                    element[10] = 8
                    continue
                elif element[10] == 'Oct':
                    element[10] = 9
                    continue
                elif element[10] == 'Nov':
                    element[10] = 10
                    continue
                elif element[10] == 'Dec':
                    element[10] = 11
                    continue
            # match with 0's and 1's
            elif i == 15 :  
                if element[15] == 'Returning_Visitor':
                    element[15] = 1
                    continue
                else : 
                    element[15] = 0
                    continue

            elif i == 16 :
                if element[16] == 'TRUE':
                    element[16] = 1
                    continue
                else:
                    element[16] = 0
                    continue

            else :
                element[i] = int(element[i])
    print(evidence_list[0])
    return (evidence_list,label_list)
    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # choosing the kneighbors classifier from  the scikit learn library
    model = KNeighborsClassifier(n_neighbors=1)
    
    # fit the model according to the training data
    model.fit(evidence,labels)
    return model
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
     # initialize the counters :
    total_right = 0
    total_false = 0
    right_right = 0 
    right_false = 0

    # we will loop through the labels and predictions together and compare every pair : 
    for label,prediction in zip(labels,predictions):
        if label == 1 :
            total_right += 1 
            if prediction == label:
                right_right += 1
        
        else:
            total_false += 1 
            if prediction == label:
                right_false += 1 

    # we compute the proportions : 
    sensitivity = float(right_right/total_right)
    specificity = float(right_false/total_false) 

    #return the tuple
    return (sensitivity,specificity)
    raise NotImplementedError


if __name__ == "__main__":
    main()
