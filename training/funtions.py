from sklearn.preprocessing import MinMaxScaler


def Scaling(train, test):
    ''' To scale the data'''
    
    sc = MinMaxScaler()
    df_train = sc.fit_transform(train)
    df_test = sc.transform(test)
    return df_train, df_test
