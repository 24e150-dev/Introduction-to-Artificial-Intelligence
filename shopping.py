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
    evidence=[]
    labels=[]
    month_map={"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5,
               "Jun":5, "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}
    with open (filename, mode='r', encoding='utf-8') as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            evidence.append([
                        int (row[0]),
                        float (row[1]),
                        int (row[2]),
                        float (row[3]),
                        int (row[4]),
                        float (row[5]),
                        float (row[6]),
                        float (row[7]),
                        float (row[8]), 
                        float (row[9]),
                        month_map [row[10]],
                        int (row[11]),
                        int (row[12]),
                        int (row [13]),
                        int (row[14]),
                        1 if row[15]=="Returning_Visitor" else 0,
                        1 if row[16].upper()=="TRUE" else 0
                    ])
            labels.append(1 if row[17].upper()=="TRUE" else 0)
    return (evidence, labels)


def train_model(evidence, labels):
    model=KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model


def evaluate(labels, predictions):
    actual_positive=0
    true_positive=0
    actual_negative=0
    true_negative=0
    for actual, predicted in zip(labels, predictions):
        if actual==1:
            actual_positive+=1
            if predicted==1:
                true_positive+=1
        else:
            actual_negative+=1
            if predicted==0:
                true_negative+=1
    sensitivity= true_positive / actual_positive if true_positive > 0 else 0.0
    specificity= true_negative / actual_negative if true_negative > 0 else 0.0
    return (sensitivity , specificity)


if __name__ == "__main__":
    main()
