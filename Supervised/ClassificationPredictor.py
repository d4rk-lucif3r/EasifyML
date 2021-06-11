accuracy_scores ={}
def predictor(
    features, labels, predictor ='lr', params={}, tune = False, test_size = .2, cv_folds =10,
    random_state =42, hidden_layers = 4, output_units = 1, input_units = 6,
    input_activation = 'relu', output_activation = 'sigmoid',optimizer = 'adam',
    metrics= ['accuracy'], loss = 'binary_crossentropy',validation_split = .20, epochs = 100
             ):
    global accuracy_scores
    """
    Encode Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2 , scales X_train, X_val using StandardScaler.
    Fits every model on training set and predicts results find and plots Confusion Matrix, 
    finds accuracy of model applies K-Fold Cross Validation 
    and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
    Applies GridSearch Cross Validation and gives best params out from param list.
    
    Parameters:
        features : array 
                    features array
                    
        lables : array
                    labels array
                    
        predictor : str
                    Predicting model to be used
                    Default 'lr'
                         Predictor Strings:
                            lr - Logisitic Regression
                            svm -SupportVector Machine
                            knn - K-Nearest Neighbours
                            dt - Decision Trees
                            nb - GaussianNaive bayes
                            rfc- Random Forest Classifier
                            xgb- XGBoost Classifier
                            ann - Artificial Neural Network
        params : dict
                    contains parameters for model
        tune : boolean  
                when True Applies GridSearch CrossValidation   
                Default is False
            
        test_size: float or int, default=.2
                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
                    include in the test split. If int, represents the absolute number of test samples. 
        cv_folds : int
                No. of cross validation folds. Default = 10
        hidden_layers : int
                No. of default layers of ann. Default = 4
        inputs_units : int
                No. of units in input layer. Default = 6
        output_units : int
                No. of units in output layer. Default = 6
        input_activation : str 
                Activation function for Hidden layers. Default = 'relu'
        output_activation : str 
                Activation function for Output layers. Default = 'sigmoid'
        optimizer: str
                Optimizer for ann. Default = 'adam'
        loss : str
                loss method for ann. Default = 'binary_crossentropy'
        validation_split : float or int
                Percentage of validation set splitting in ann. Default = .20
        epochs : int
                No. of epochs for ann. Default = 100
                
  
    
    EX: 
      For Logistic Regression
            predictor
                (
                    features = features, 
                    labels = labels, 
                    predictor = 'lr', 
                    {'penalty': 'l1', 'solver': 'liblinear'}, 
                    tune = True, 
                    test_size = .25
                )
    
    """
    print('Checking if labels or features are categorical! [*]\n')
    cat_features=[i for i in features.columns if features.dtypes[i]=='object']
    if len(cat_features) >= 1 :
        index = []
        for i in range(0,len(cat_features)):
            index.append(features.columns.get_loc(cat_features[i]))
        print('Features are Categorical\n')
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), index)], remainder='passthrough')
        print('Encoding Features [*]\n')
        features = np.array(ct.fit_transform(features))
        print('Encoding Features Done [',u'\u2713',']\n')
    if labels.dtype == 'O':
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        le = LabelEncoder()
        print('Labels are Categorical [*] \n')
        print('Encoding Labels \n')
        labels = le.fit_transform(labels)
        print('Encoding Labels Done [',u'\u2713',']\n')
    else:
        print('Features and labels are not categorical [',u'\u2713',']\n')

    print('Applying SMOTE [*]\n')
    from imblearn.over_sampling import SMOTE
    sm=SMOTE(k_neighbors=4)
    features,labels=sm.fit_resample(features,labels)
    print('SMOTE Done [',u'\u2713',']\n')
    
    print('Splitting Data into Train and Validation Sets [*]\n')
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size= test_size, random_state= random_state)
    print('Splitting Done [',u'\u2713',']\n')
    
    print('Scaling Training and Test Sets [*]\n')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    print('Scaling Done [',u'\u2713',']\n')
    
     
    if predictor == 'lr':
        print('Training Logistic Regression on Training Set [*]\n')
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(**params)
        parameters= [{
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C' : np.logspace(-4, 4, 20),
        }]

    elif predictor == 'svm':
        print('Training Support Vector Machine on Training Set [*]\n')
        from sklearn.svm import SVC
        classifier = SVC(**params)
        parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C' : np.logspace(-4, 4, 20)},
            {'kernel': ['linear'],'gamma': [1e-3, 1e-4,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C' : np.logspace(-4, 4, 20)},
            {'kernel': ['poly'], 'gamma': [1e-3, 1e-4,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C' : np.logspace(-4, 4, 20)},
            {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C' : np.logspace(-4, 4, 20)},
                     ]
    elif predictor == 'knn':
        print('Training K-Nearest Neighbours on Training Set [*]\n')
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(**params)
        parameters = [{
            'n_neighbors': list(range(0,31)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_jobs': [1, 0, None]
        }]

    elif predictor == 'dt':
        print('Training Decision Tree Classifier on Training Set [*]\n')
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(**params)
        parameters= [{
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': [2, 3],
            'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],

        }]
    elif predictor == 'nb':
        print('Training Naive Bayes Classifier on Training Set [*]\n')
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(**params)
        
    elif predictor == 'rfc':
        print('Training Random Forest Classifier on Training Set [*]\n')
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(**params)
        parameters = [{
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
            'bootstrap': [True,False],
            'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
        }]
    elif predictor == 'xgb':
        print('Training XGBClassifier on Training Set [*]\n')
        from xgboost import XGBClassifier
        classifier = XGBClassifier(**params)
        parameters = {
            'min_child_weight': [1, 5, 10],
            'gamma': [1e-3, 1e-4,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
            'learning_rate': [0.3, 0.1, 0.03],
        }
    elif predictor =='ann':
        print('Training ANN on Training Set [*]\n')
        import tensorflow as tf
        classifier = tf.keras.models.Sequential()
        for i in range(0,hidden_layers):
            classifier.add(tf.keras.layers.Dense(units = input_units, activation = input_activation))
        classifier.add(tf.keras.layers.Dense(units = output_units, activation = output_activation))
        classifier.compile(optimizer = optimizer, loss = loss, metrics = metrics,)
        ann_history = classifier.fit(X_train, y_train,validation_split = validation_split , validation_data = (X_val, y_val), epochs = epochs)
        accuracy_scores[predictor] = classifier.evaluate(X_train, y_train)[1] * 100
#         ann_history.history['accuracy']
        return ann_history
    if not predictor == 'ann': 
        classifier.fit(X_train, y_train)
    print('Model Training Done [',u'\u2713',']\n')
                              
    print('''Making Confusion Matrix [*]''')
    from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
    y_pred = classifier.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print('Confusion Matrix Done [',u'\u2713',']\n')
    
    plot_confusion_matrix(classifier, X_val, y_val, cmap="pink")
    
    print('''Evaluating Model Performance [*]''')
    accuracy = accuracy_score(y_val, y_pred)
    print('Validation Accuracy is :',accuracy)
    print('Evaluating Model Performance [',u'\u2713',']\n')
    
    print('Applying K-Fold Cross validation [*]')
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=cv_folds,)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    accuracy_scores[classifier] = accuracies.mean()*100
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))   
    print('K-Fold Cross validation [',u'\u2713',']\n')
    if not predictor == 'nb' and tune and predictor =='ann':
        print('Applying Grid Search Cross validation [*]')
        from sklearn.model_selection import GridSearchCV,StratifiedKFold
        
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=parameters,
            scoring='accuracy',
            cv=cv_folds,
            n_jobs=-1,
            verbose=4,
        )
        grid_search.fit(X_train, y_train)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
        print("Best Parameters:", best_parameters)
        print('Applying Grid Search Cross validation [',u'\u2713',']\n')
        
    print('Complete [',u'\u2713',']\n')
