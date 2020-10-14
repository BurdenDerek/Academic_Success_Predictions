def machine_learning_summary(show=False):
    
    import warnings
    warnings.filterwarnings('ignore')

    import os
    import numpy as np
    import pandas as pd
    
    from pyspark.sql import SparkSession
    from pyspark import SparkFiles

    from collections import Counter

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.under_sampling import ClusterCentroids
    from imblearn.combine import SMOTEENN
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import balanced_accuracy_score
    from imblearn.metrics import classification_report_imbalanced
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.ensemble import EasyEnsembleClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint

    def extract(url, file, show):
        # takes in a AWS S3 url, file_name, and if we want to display the data after it comes in
        spark = SparkSession.builder.appName("Project_ETL").config("spark.driver.extraClassPath","/content/postgresql-42.2.9.jar").getOrCreate()
        spark.sparkContext.addFile(url)
        df = spark.read.csv(SparkFiles.get(file), sep=",", header=True, inferSchema=True)
        df = df.toPandas()

        if show == True:
            display(df.head())

        return df

    # file names
    math_file = "student-mat.csv"
    por_file = "student-por.csv"
    # file urls
    math_url = f"https://burdenderek-project.s3.us-east-2.amazonaws.com/{math_file}"
    por_url = f"https://burdenderek-project.s3.us-east-2.amazonaws.com/{por_file}"
    # save the dataframes
    math = extract(url=math_url, file=math_file, show=show)
    por = extract(url=por_url, file=por_file, show=show)

    # clean bucket the grades
    # 10 and above is a pass
    # 9 and below is a fail

    def encode_grades(data, show=False):
        # bucket the grades into passing(1) and failling(0)

        # math
        # failling
        data.loc[(data["G1"] < 10), "G1"] = 0
        data.loc[(data["G2"] < 10), "G2"] = 0
        data.loc[(data["G3"] < 10), "G3"] = 0

        # passing
        data.loc[(data["G1"] >= 10), "G1"] = 1
        data.loc[(data["G2"] >= 10), "G2"] = 1
        data.loc[(data["G3"] >= 10), "G3"] = 1

        if show == True:
            display(data.head())

        return

    encode_grades(math, show=show)
    encode_grades(por, show=show)

    dnn_math = math
    dnn_por = por

    def encode_features(data, show=False):

        for i in data.columns.tolist():
            le = LabelEncoder()
            data[i] = le.fit_transform(data[i])

        if show == True:
            display(data.head())

        return

    encode_features(math, show)
    encode_features(por, show)

    def over_sample(df, drop, target):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        # oversample to make up for the low number of risky loans
        ros = RandomOverSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        model = LogisticRegression(solver='lbfgs', random_state=1)
        model.fit(X_resampled, y_resampled)
        y_predictions = model.predict(X_test)

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        return acc_score*100

    def under_sample(df, drop, target):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        ros = RandomUnderSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        model = LogisticRegression(solver='lbfgs', random_state=1)
        model.fit(X_resampled, y_resampled)
        y_predictions = model.predict(X_test)

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        return acc_score*100

    def cluster(df, drop, target):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        cc = ClusterCentroids(random_state=1)
        X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

        model = LogisticRegression(solver='lbfgs', random_state=1)
        model.fit(X_resampled, y_resampled)

        y_predictions = model.predict(X_test)

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        return acc_score*100

    def smoteen(df, drop, target):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        smote_enn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        model = LogisticRegression(solver='lbfgs', random_state=1)
        model.fit(X_resampled, y_resampled)
        y_predictions = model.predict(X_test)

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        return acc_score*100

    def random_forest(df, drop, target, show, model_name):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
        brf.fit(X_train, y_train)
        y_predictions = brf.predict(X_test)

        feature_importance = sorted(zip(brf.feature_importances_, X.columns.tolist()))[::-1]

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        # Displaying results
        if show == True:
            print(f"Feature Importance: {model_name}")
            for i in feature_importance:
                print(i)
            print("\n")

        return acc_score*100

    def easy_ensemble_classifier(df, drop, target):

        # split the table into features and outcomes
        x_cols = [i for i in df.columns if i not in drop]
        X = df[x_cols]
        y = df[target]

        # split features and outcomes into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        eec = EasyEnsembleClassifier(n_estimators=100, random_state=0)
        eec.fit(X_train, y_train)
        y_predictions = eec.predict(X_test)

        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_predictions)

        return acc_score*100

    def dnn(df, drop, target, file_name):

        # Generate our categorical variable list
        encode_cat = df.dtypes[df.dtypes == "object"].index.tolist()

        # Check the number of unique values in each column
        df[encode_cat].nunique()

        # Create the OneHotEncoder instance
        enc = OneHotEncoder(sparse=False)

        # Fit the encoder and produce encoded DataFrame
        encode_df = pd.DataFrame(enc.fit_transform(df[encode_cat]))

        # Rename encoded columns
        encode_df.columns = enc.get_feature_names(encode_cat)

        # Merge the two DataFrames together and drop the Country column
        df = df.merge(encode_df,left_index=True,right_index=True).drop(encode_cat, 1)

        # Split our preprocessed data into our features and target arrays
        y = df[target].values
        X = df.drop(drop,1).values

        # Split the preprocessed data into a training and testing dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

        # Create a StandardScaler instance
        scaler = StandardScaler()

        # Fit the StandardScaler
        X_scaler = scaler.fit(X_train)

        # Scale the data
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        # Define the model - deep neural net
        number_input_features = len(X_train[0])
        hidden_nodes_layer1 =  len(X_train[0]) * 2
        hidden_nodes_layer2 = len(X_train[0]) * .1

        nn = tf.keras.models.Sequential()

        # First hidden layer
        nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

        # Second hidden layer
        nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

        # Output layer
        nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

        # Compile the model
        nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train the model
        fit_model = nn.fit(X_train_scaled,y_train,epochs=50, verbose=0)

        # We are going to do a slightly round about method to test our model
        # We are saving and exporting the model then importing it back in
        # This is for two reasons
        # First reason is that we want to save our trained models
        # But we do not need it reimport it to test its accuracy, so why are we doing this?
        # We are testing the imported model because we want to make sure that the model file works

        # save model
        nn.save(file_name)

        # import model back in
        nn_imported = tf.keras.models.load_model(file_name)

        # Evaluate the model using the test data
        model_loss, model_accuracy = nn_imported.evaluate(X_test_scaled,y_test, verbose=0)

        return model_accuracy*100

    def model_summary(df, drop, target, model_name, show, file_name, dnn_df):

        # make a dataframe to neatly organize our results
        machine_learning_summary = pd.DataFrame(
            {
                "Target": model_name,
                "Over Sampling": [over_sample(df, drop, target)],
                "Under Sampling": [under_sample(df, drop, target)],
                "Cluster Centroids": [cluster(df, drop, target)],
                "SMOTEENN": [smoteen(df, drop, target)],
                "Random Forest": [random_forest(df, drop, target, show, model_name)],
                "Easy Ensemble Classifier": [easy_ensemble_classifier(df, drop, target)],
                "Deep Neural Network": [dnn(dnn_df, drop, target, file_name)]
            }
        )

        # format the accuracy scores to make them easier to read and more descriptive
        machine_learning_summary["Over Sampling"] = machine_learning_summary["Over Sampling"].map("{:.1f}%".format)
        machine_learning_summary["Under Sampling"] = machine_learning_summary["Under Sampling"].map("{:.1f}%".format)
        machine_learning_summary["Cluster Centroids"] = machine_learning_summary["Cluster Centroids"].map("{:.1f}%".format)
        machine_learning_summary["SMOTEENN"] = machine_learning_summary["SMOTEENN"].map("{:.1f}%".format)
        machine_learning_summary["Random Forest"] = machine_learning_summary["Random Forest"].map("{:.1f}%".format)
        machine_learning_summary["Easy Ensemble Classifier"] = machine_learning_summary["Easy Ensemble Classifier"].map("{:.1f}%".format)
        machine_learning_summary["Deep Neural Network"] = machine_learning_summary["Deep Neural Network"].map("{:.1f}%".format)

        # change the index name it more clearly state that it is the accuracy scores being displayed
        # machine_learning_summary = machine_learning_summary.rename(index={0: "Accuracy Score"})
        machine_learning_summary = machine_learning_summary.set_index("Target")
        # show us the dataframe
        #display(machine_learning_summary)

        return machine_learning_summary

    def accuracy_score_table(show=False):

        # different columns to drop depending on which target we are using
        # we are not dropping previous grades because it is a reasonable expectation to have those data points sequential trimesters
        G1 = ["G1", "G2", "G3"]
        G2 = ["G2", "G3"]
        G3 = ["G3"]

        # names from our different model targets
        models = [
            "Math G1",
            "Math G2",
            "Math G3",
            "Portuguese G1",
            "Portuguese G2",
            "Portuguese G3"
        ]

        # names for the different file names
        file = [
            "trained_math_G1.h5",
            "trained_math_G2.h5",
            "trained_math_G3.h5",
            "trained_por_G1.h5",
            "trained_por_G2.h5",
            "trained_por_G3.h5"
        ]

        summary_table = model_summary(df=math, drop=G1, target="G1", model_name=models[0], show=show, file_name=file[0], dnn_df=dnn_math)
        summary_table = summary_table.append(model_summary(df=math, drop=G2, target="G2", model_name=models[1], show=show, file_name=file[1], dnn_df=dnn_math))
        summary_table = summary_table.append(model_summary(df=math, drop=G3, target="G3", model_name=models[2], show=show, file_name=file[2], dnn_df=dnn_math))
        summary_table = summary_table.append(model_summary(df=por, drop=G1, target="G1", model_name=models[3], show=show, file_name=file[3], dnn_df=dnn_por))
        summary_table = summary_table.append(model_summary(df=por, drop=G2, target="G2", model_name=models[4], show=show, file_name=file[4], dnn_df=dnn_por))
        summary_table = summary_table.append(model_summary(df=por, drop=G3, target="G3", model_name=models[5], show=show, file_name=file[5], dnn_df=dnn_por))

        display(summary_table)

        return

    accuracy_score_table(show)

    return