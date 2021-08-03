import cv2 as cv
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import Perceptron, Lasso

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC


def img_to_bp(img):
    w, h = 100-2, 100-2
    x0, y0 = 8, 6
    gap = 10
    n = 0
    dict_img_cells = {}
    for k in range(2):
        for i in range(3):
            for j in range(2):
                x, y = x0+293*k+(w+gap)*j, y0+(h+gap)*i
                cropped_img = img[y:y+h, x:x+w]
                dict_img_cells[n] = cropped_img
                n += 1
    return dict_img_cells


def get_centroid(moments):
    cx, cy = moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]
    return [cx, cy]


def get_circularity(contour_area, hull_area, perimetro, perimetro_hull):
    return [4*np.pi*contour_area/(perimetro**2), 4*np.pi*hull_area/(perimetro_hull**2)]


def get_ellipticity(H1, H2):
    try:
        return (1/(2*(np.pi**2)))*(1/(H1*np.sqrt(4*H2+(1/(np.pi**2)))-2*H2))
    except Exception:
        return 0


def get_inclination(nu20, nu02):
    return (nu20)/(10**(-12)+nu02)


def get_orientation(mu20, mu11, mu02):
    try:
        r = 2 * mu11 / (mu20-mu02)
        return (1/2) * np.arctan(r)
    except Exception:
        return 0


def get_eccentricity(m00, mu20, mu11, mu02):
    try:
        mu20_ = mu20/m00
        mu02_ = mu02/m00
        mu11_ = mu11/m00
        r0 = (mu20_+mu02_) / 2
        r1 = 4*(mu11_**2) + (mu20_-mu02_)**2
        l1 = r0 + np.sqrt(r1)/2
        l2 = r0 - np.sqrt(r1)/2
        return np.sqrt(1-l2/l1)
    except Exception:
        return 0


columns_ = ["masa", "hull size", "size", "area contorno", "area envoltura convexa", "perimetro contorno",
            "perimetro envoltura convexa", "ratio area", "ratio perimetro", "centroide x", "centroide y",
            "circularity contour", "circularity hull", "ellipticity", "inclination", "orientation", "eccentricity",
            "centrality x", "centrality y", "skewness x", "skewness y", "H1", "H2", "H3", "H4"]


def get_rep_vect(bp, n):
    img = bp[n]
    img[img > 175] = 255
    img[img < 175] = 0
    masa = img[img > 175].size
    if masa > 0:
        moments = cv.moments(img)
        hu_moments = np.array(cv.HuMoments(moments)).flatten()[0:4]
        tr_moments = np.sign(-hu_moments)*np.log(abs(hu_moments))
        centroid = get_centroid(moments)
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv.contourArea)
        hull = cv.convexHull(contour)
        ellipticity = get_ellipticity(tr_moments[0], tr_moments[1])
        inclination = get_inclination(moments["nu20"], moments["nu02"])
        orientation = get_orientation(moments["mu20"], moments["mu11"], moments["mu02"])
        eccentricity = get_eccentricity(moments["m00"], moments["mu20"], moments["mu11"], moments["mu02"])
        centrality = moments["nu12"], moments["nu21"]
        skewness = moments["nu30"], moments["nu03"]
        contour_area = cv.contourArea(contour)
        hull_area = cv.contourArea(hull)
        perimetro_hull = cv.arcLength(hull, True)
        perimetro = cv.arcLength(contour, True)
        circularity = get_circularity(contour_area, hull_area, perimetro, perimetro_hull)
        otros = [masa, hull.size, contour_area, contour_area/masa, hull_area/masa,
                 perimetro/masa, perimetro_hull/masa, contour_area/hull_area,
                 perimetro/perimetro_hull]
        cols = np.concatenate((otros, centroid, circularity, [ellipticity], [inclination], [orientation],
                               [eccentricity], centrality, skewness, tr_moments))
        return cols
    else:
        return [0 for _ in range(len(columns_))]


def get_X_y(bp_id, n_pca=0, n_select=0, alpha_lasso=0, n_lasso=1, transform=True):
    X = [get_rep_vect(bp_id, n) for n in range(12)]
    y = np.arange(12) < 6
    if transform:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    if n_pca > 0:
        pca_ = PCA(n_components=n_pca)
        X = pca_.fit_transform(X)
        df = pd.DataFrame(X)
    elif n_select > 0:
        df = pd.DataFrame(X, columns=columns_)
        reducer = SelectKBest(f_classif, k=n_select).fit(df, y)
        df = reducer.transform(df)
        col = np.array(columns_)[reducer.get_support(True)]
        df = pd.DataFrame(df, columns=col)
        X = df.copy()
    elif alpha_lasso > 0:
        df = pd.DataFrame(X, columns=columns_)
        lasso = Lasso(alpha=alpha_lasso)
        lasso.fit(X, y)
        n_best = np.argpartition(abs(lasso.coef_), -n_lasso)[-n_lasso:]
        col = [columns_[i] for i in n_best if abs(lasso.coef_[i]) > 0]
        df = pd.DataFrame(df, columns=col)
        X = df.copy()
    else:
        df = pd.DataFrame(X, columns=columns_)
    df["label"] = y
    return df, X, y


def split_train_test_ij(df, i, j):
    _train, _test = df.drop([i, j], axis=0), df.loc[[i, j]]
    X_train, X_test = _train.drop(["label"], axis=1), _test.drop(["label"], axis=1)
    y_train, y_test = _train["label"], _test["label"]
    return _train, _test, X_train, X_test, y_train, y_test


def performance_models(df, n_dt=1):
    data_bp = {"pair_ij":[], "perceptron":[], "svm":[], "NB":[], "dummy": [], "Decision-Tree": []}
    for i in range(6):
        for j in range(6, 12):
            # train-test
            _train, _test, X_train, X_test, y_train, y_test = split_train_test_ij(df, i, j)
            # Perceptron
            clf_perceptron = Perceptron(tol=1e-6, random_state=0)
            clf_perceptron.fit(X_train, y_train)
            score_perceptron_test = clf_perceptron.score(X_test, y_test)
            # SVM
            clf_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', tol=1e-6))
            clf_svm.fit(X_train, y_train)
            score_svm_test = clf_svm.score(X_test, y_test)
            # NB-Gaussian
            clf_NB = GaussianNB()
            clf_NB.fit(X_train, y_train)
            score_NB_test = clf_NB.score(X_test, y_test)
            # Dummy
            clf_dummy = DummyClassifier(strategy="uniform")
            clf_dummy.fit(X_train, y_train)
            score_dummy_test = clf_dummy.score(X_test, y_test)
            # Decision-Tree
            clf_Dtree = DecisionTreeClassifier(max_depth=n_dt, random_state=1)
            clf_Dtree.fit(X_train, y_train)
            score_Dtree_test = clf_Dtree.score(X_test, y_test)
            data_bp["pair_ij"].append(f"{i}|{j}")
            data_bp["perceptron"].append(np.floor(score_perceptron_test))
            data_bp["svm"].append(np.floor(score_svm_test))
            data_bp["NB"].append(np.floor(score_NB_test))
            data_bp["dummy"].append(np.floor(score_dummy_test))
            data_bp["Decision-Tree"].append(np.floor(score_Dtree_test))
    df_bp = pd.DataFrame(data_bp)
    df_bp = df_bp.set_index("pair_ij")
    return df_bp


def performance_per_cell(df_bp):
    cols = df_bp.columns
    dict_cols = {col: [] for col in cols}
    for i in range(12):
        indexs_i = [ix for ix in df_bp.index if str(i) in ix.split("|")]
        for col in cols:
            col_pf = 0
            for ix in indexs_i:
                col_pf += int(df_bp.loc[str(ix)][col])
            dict_cols[col].append(np.round(col_pf/len(indexs_i), 3))
    return pd.DataFrame(dict_cols)


def solve():
    filename = '..\API\some_image.jpg'
    img = ~cv.imread(filename, 0)
    bp = img_to_bp(img)
    df, X, y = get_X_y(bp, n_pca=0, n_select=1, alpha_lasso=0, n_lasso=0, transform=True)
    df_bp = performance_models(df, n_dt=1)
    output = {}
    output["models"] = df_bp.mean().to_dict()
    output["solution"] = list(df.columns)[:-1]
    return output


if __name__ == "__main__":
    print(solve())