from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame({
    'Hours': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5],
    'Result': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]})
df

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# Find the formula for the polynomial regression of nth degree.
# When you pass a dictionary whose key value is non-negative integer into `formula`, it gives you the formula as follows:
# y = dictionary(0) + dictionary(1) x^1 + dictionary(2) x^2 + ... dictionary (n) x^n 
def formula(func):
    def unpacking(*args, **kwargs):
        result = ""
        test_dic = func(*args, **kwargs)
        for key in test_dic:
            if key == 0:
                result = result + str(round(test_dic.get(key), 4))
            else:
                if test_dic.get(key) >= 0:
                    result = result + " + " + str(round(test_dic.get(key), 4)) + f" x^{key}"
                else:
                    result = result + " + " + "(" + str(round(test_dic.get(key), 4)) + ")" + f" x^{key}"
        return "y = " + result    
    return unpacking

# Poly_regression is a function which convert regressor coefficients and regressor intercept into a dictionary.
@formula
def poly_regression(degree=2):
    polyreg = PolynomialFeatures(degree, include_bias=False)
    X_poly = polyreg.fit_transform(X)

    regressor = LinearRegression()
    regressor.fit(X_poly, y)

    coef_list = list(regressor.coef_)
    coef_list.insert(0, regressor.intercept_)   
    
    index_list = list(range(len(coef_list)))
    as_dict = dict(zip(index_list, coef_list))

    
    return as_dict



# Plot the polynomial regression of nth degree when n is less than eleven.
#dic_for_degree is for the title of plots.



def poly_graph(degree=2):
    polyregpipe=make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    polyregpipe.fit(X,y)

    dic_for_degree = {
    2 : "second",
    3 : "third",
    4 : "fourth",
    5 : "fifth",
    6 : "sixth",
    7 : "seventh",
    8 : "eighth",
    9 : "nineth",
    10 : "tenth"
}
    
    plt.scatter(X,y)
    plt.plot(X, polyregpipe.predict(X))
    plt.title(f"Polynomial regression of the {dic_for_degree.get(degree)} degree")
    plt.show()
    return plt.figure()

