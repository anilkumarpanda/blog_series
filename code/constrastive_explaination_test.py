# Code to explore constrastive explaination.
# Compare this with Shap explaination.

import pandas as pd
from loguru import logger
from blog.data.data_cleaner_factory import DataCleanerFactory
import contrastive_explanation as ce
from sklearn import datasets, model_selection, ensemble, metrics

dcf = DataCleanerFactory()
lnt_dataset = dcf.getDataset("lnt")
X, y = lnt_dataset.get_data(path="data/lnt_dataset.csv")
seed = 1

print(type)
train, test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.70, random_state=seed
)
print(type(train))
print(type(test))
print(type(y_train))
print(type(y_test))

model = ensemble.RandomForestClassifier(random_state=seed)
model.fit(train, y_train)

# Print out the classifier performance (F1-score)
print(
    "Classifier performance (F1):",
    metrics.f1_score(y_test, model.predict(test), average="weighted"),
)

dm = ce.domain_mappers.DomainMapperPandas(train, contrast_names=["0" "1"])
exp = ce.ContrastiveExplanation(dm, verbose=True)

sample = test.iloc[4]
print(sample)
print(exp.explain_instance_domain(model.predict_proba, sample))
