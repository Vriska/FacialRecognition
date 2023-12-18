import xgboost as xgb
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
import joblib
results = joblib.load(r"C:\\Users\Vriska S\Desktop\MMS Project\data.pkl")
new, target = results[0],results[1]
param_grid = {'C': [1, 2, 3, 4, 5],
              'gamma': [0.5, 1, 2, 4, 8],
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(new, target)
print(grid.best_params_)

#clf = xgb.XGBClassifier(eta=0.5,objective='multi:softmax')
clf = SVC(probability=True, kernel='linear', gamma= 1,C=2)
clf.fit(new,target)
joblib.dump(clf,"C:\\Users\Vriska S\Desktop\MMS Project\dumpnewardu.pkl")
print("done")
