# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FinalUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QMessageBox


class Ui_MainWindow(QDialog):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 370)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #Linear Regression
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(70, 160, 121, 17))
        self.radioButton.setObjectName("radioButton")
        #####Function call for radio_button selection in the main screen
        self.radioButton.clicked.connect(lambda: self.MultipleLinearReg())
        #####
        
        #Support Vector Regression
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(70, 180, 171, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        #####Function call for radio_button2 selection in the main screen
        self.radioButton_2.clicked.connect(lambda: self.SupportVectorRegression())
        #####
        
        #Decision Tree Algorithm
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(70, 200, 171, 17))
        self.radioButton_3.setObjectName("radioButton_3")
        #####Function call for radio_button3 selection in the main screen
        self.radioButton_3.clicked.connect(lambda: self.DecisionTree())
        #####
        
        
        #Random Forest Regression
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_4.setGeometry(QtCore.QRect(70, 220, 161, 17))
        self.radioButton_4.setObjectName("radioButton_4")
        #####Function call for radio_button4 selection in the main screen
        self.radioButton_4.clicked.connect(lambda: self.RandomForestReg())
        #####
        
        #Label = What is the best regression?
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 100, 211, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        #Logistic Regression Classifier
        self.radioButton_5 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_5.setGeometry(QtCore.QRect(70, 240, 161, 17))
        self.radioButton_5.setObjectName("radioButton_5")
        #####Function call for radio_button5 selection in the main screen
        self.radioButton_5.clicked.connect(lambda: self.PolyRegression())
        #####
        
        #Output(Label_2)
        #This is unused variable
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(280, 150, 171, 111))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

##################################
#GUI widget screen setting domain
##################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
##################################
#GUI text domain
##################################
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton.setText(_translate("MainWindow", "Linear Regression"))
        self.radioButton_2.setText(_translate("MainWindow", "Support Vector Regression"))
        self.radioButton_3.setText(_translate("MainWindow", "Decision Tree Algorithm"))
        self.radioButton_4.setText(_translate("MainWindow", "Random Forest Regression"))
        self.label.setText(_translate("MainWindow", "What is the Best Regression?"))
        self.radioButton_5.setText(_translate("MainWindow", "Logistic Regression Classifier"))
        self.label_2.setText(_translate("MainWindow", "Output"))
                
            
##################################      
#
####Function1
####Multiple Linear Regression
#
##################################     

    def MultipleLinearReg(self):
        
        # Importing the libraries 
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        #Importing the dataset
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        #reshape y
        y = y.reshape(len(y),1)
            
        #Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        #Training the Multiple Linear Regression model on the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        #Predicting the Test set results
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        
        #Evaluating the Model Performance
        from sklearn.metrics import r2_score
        r2_MultLinearReg = r2_score(y_test, y_pred)
        print("R-squared score of Multiple Linear Regression Model is:",r2_MultLinearReg)
        
        ##################################
        #Msgbox setting
        msg = QMessageBox()
        msg.setWindowTitle("Best R2 result")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("R-squared score of Multiple Linear Regression Model is:")
        msg.setText(str(r2_MultLinearReg))
        msg.exec()
        ##################################
        
##################################      
#
####Function2
####Support Vector Regression
#
##################################  

    def SupportVectorRegression(self):
       
        # Importing the libraries 
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        #Importing the dataset
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        y = y.reshape(len(y),1)
            
        #Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        y_train = sc_y.fit_transform(y_train)
        
        #Training the SVR model on the Training set
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        
        #Predicting the Test set results
        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
        np.set_printoptions(precision=2)
        
        #Evaluating the Model Performance
        from sklearn.metrics import r2_score
        r2_score(y_test, y_pred)
        r2_SuppVecReg = r2_score(y_test, y_pred)
        print("R-squared score of Support Vector Regression Model is:",r2_SuppVecReg) 
        
        ##################################
        #Msgbox setting
        msg = QMessageBox()
        msg.setWindowTitle("Best R2 result")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("R-squared score of Support Vector Regression Model is:")
        msg.setText(str(r2_SuppVecReg))
        msg.exec()
        ##################################
        
##################################      
#
####Function3
####Decision Tree Algorithm
#
##################################    
    def DecisionTree(self):
    
        # Importing the libraries 
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        #Importing the dataset
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
            
        #Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
      
        #Training the Decision Tree Regression model on the Training set
    
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X_train, y_train)
        
        #Predicting the Test set results
    
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        #Evaluating the Model Performance
        from sklearn.metrics import r2_score
        r2_DecTree = r2_score(y_test, y_pred);
        
        #Print Decision Tree R2 Score Output
        print("R-squared score of Decision Tree Model is:",r2_DecTree)  
        
        ##################################
        #Msgbox setting
        msg = QMessageBox()
        msg.setWindowTitle("Best R2 result")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("R-squared score of Decision Tree Model is:")
        msg.setText(str(r2_DecTree))
        msg.exec()
        ##################################
        
##################################      
#
####Function4
####Random Forest Regression
#
##################################   
    def RandomForestReg(self):
        
        # Importing the libraries 
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        #Importing the dataset
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        #Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        #Training the Random Forest Regression model on the whole dataset¶
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train, y_train)
        
        #Predicting the Test set results
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        
        #Evaluating the Model Performance
        from sklearn.metrics import r2_score
        r2_score(y_test, y_pred)
        r2_RandomForReg = r2_score(y_test, y_pred)
        print("R-squared score of Random Forest Regression Model is:",r2_RandomForReg )
        
        ##################################
        #Msgbox setting
        msg = QMessageBox()
        msg.setWindowTitle("Best R2 result")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("R-squared score of Random Forest Regression Model is:")
        msg.setText(str(r2_RandomForReg))
        msg.exec()
        ##################################

##################################      
#
####Function5
####Logistic Regression Classifier(Polynomial Regression)
#
##################################   
    def PolyRegression(self):
        
        # Importing the libraries 
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        #Importing the dataset
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
            
        #Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        #Training the Polynomial Regression model on the Training set
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, y_train)
        
        #Predicting the Test set results
        y_pred = regressor.predict(poly_reg.transform(X_test))
        np.set_printoptions(precision=2)
        
        #Evaluating the Model Performance
        from sklearn.metrics import r2_score
        r2_score(y_test, y_pred)
        r2_polyreg = r2_score(y_test, y_pred)
        print("R squared score of Polynomial Regression Model is:",r2_polyreg)
        
        ##################################
        #Msgbox setting
        msg = QMessageBox()
        msg.setWindowTitle("Best R2 score result")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("R-squared score of Polynomial Regression Model is:")
        msg.setText(str(r2_polyreg))
        msg.exec()
        ##################################

##################################
#Main domain window
##################################
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

