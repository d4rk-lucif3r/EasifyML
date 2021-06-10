def detect_outliers(dataset, columns):
    """
    This function takes dataset and columns as input and finds Q1, Q3 and IQR for that list of column
    Detects the outlier and it index and stores them in a list.
    Then it creates as counter object with that list and stores it
    in Multiple Outliers list if the value of outlier is greater than 1.5
    
    Ex:
    1) For printing no. of outliers.
      print("number of outliers detected --> ",
      len(dataset.loc[detect_outliers(dataset, dataset.columns[:-1])]))
    2) Printing rows and columns collecting the outliers
      dataset.loc[detect_outliers(dataset.columns[:-1])]
    3) Dropping those detected outliers
      dataset = dataset.drop(detect_outliers(dataset.columns[:-1]),axis = 0).reset_index(drop = True)
    
    
    
    """
    outlier_indices = []
    for column in columns:
        # 1st quartile
        Q1 = np.percentile(dataset[column], 25)
        # 3st quartile
        Q3 = np.percentile(dataset[column], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier Step
        outlier_step = IQR * 1.5
        # detect outlier and their indices
        outlier_list_col = dataset[(dataset[column] < Q1 - outlier_step)
                              | (dataset[column] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1.5)

    return multiple_outliers
