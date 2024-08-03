class ModelTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._get_model()

    def _get_model(self):
        models = {
            'MLP': MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
        }
        return models[self.model_name]

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
