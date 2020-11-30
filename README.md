<h1>GML <img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence9-512.png" alt="Brain+Machine" height="38" width="38"> </img> <img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence15-512.png" alt="Adding AI" height="38" width="38"> </img> <img src="https://cdn1.iconfinder.com/data/icons/science-technology-outline/91/Science__Technology_35-512.png" alt="Revolution" height="38" width="38"> </img>  </h1>

[![Generic badge](https://img.shields.io/badge/Data_Science-AUTO-<COLOR>.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning)
[![Generic badge](https://img.shields.io/badge/Machine_Learning-AUTO-<COLOR>.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning) 
[![Generic badge](https://img.shields.io/badge/Deep_Learning-AUTO-<COLOR>.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning)
[![Generic badge](https://img.shields.io/badge/NLP-AUTO-<COLOR>.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning)<br>
[![PyPI version](https://badge.fury.io/py/GML.svg)](https://pypi.org/project/GML)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.org/project/GML/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.org/project/GML/)
[![GitHub issues](https://img.shields.io/github/issues/Muhammad4hmed/Ghalat-Machine-Learning)](https://GitHub.com/Muhammad4hmed/Ghalat-Machine-Learning/issues/)


<h2> Creators </h2>
<a href="https://www.linkedin.com/in/muhammad4hmed/">Muhammad Ahmed</a> <br>
<a href="https://www.linkedin.com/in/naman-tuli-3213361a6">Naman Tuli</a>

<h2> Contributors </h2>
<a href="https://www.linkedin.com/in/rafeyirahman">Rafey Iqbal Rahman</a> 
<br>
<br>

<b>Tired of doing Data Science manually? GML is here for you!</b>
<br>
<br>
GML is an automatic data science library in python built on top of multiple Python packages. Complete features which we offer are listed as: <br>
<img src="https://i.ibb.co/L1mpQR1/Untitled-design-High-Quality-3.jpg">
<br>
<br>
<h2>Installation: </h2> <br>

```python
pip install GML
```

<br>
<a href = "https://pypi.org/project/GML/">https://pypi.org/project/GML</a> 
<br>
<br>
<h2>Features:</h2><br>
<h3>Auto Feature Engineering</h3> <br>
<br>

```python
from GML import FeatureEngineering

fe = FeatureEngineering(Data, 'target', fill_missing_data=True, encode_data=True, 
                        normalize=True, remove_outliers=True, 
                        new_features=True, feateng_steps=2 ) # feateng_steps = 0 for features selection without feature creation

X_new, y, test = fe.get_new_data()
```

<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/FeatureEngineering.ipynb">Here</a> for complete DEMO</p>
<br>
<h3>Auto EDA (Powered by Sweetviz)</h3> <br>
<br>

```python
from GML import sweetviz

result1 = sweetviz.compare([train,'train'],[test,'test'],'target') 
result2 = sweetviz.analyze([train,'train'])

result.show_html()
result2.show_html()
```

<a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/GML_ANALYZE_REPORT.html"> <img src="https://i.ibb.co/wgzQfgy/Screenshot-2020-11-30-Screenshot.png"> </a>
<a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/GML_COMPARE_REPORT.html"> <img src="https://i.ibb.co/0BpHYJZ/Screenshot-2020-11-30-Screenshot-1.png"> </a>
<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/AutoEDA.ipynb">Here</a> for complete DEMO</p>
<br>                             
<h3> Auto Machine Learning </h3> <br>
<br>

```python
from GML import AutoML

gml_ml = AutoML()

gml_ml.GMLClassifier(X, y, metric = accuracy_score, folds = 10)
```

<br>
<img src="https://i.ibb.co/s3x77XZ/Screenshot-2020-11-30-Auto-Machine-Learning-Jupyter-Notebook.png">
<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/AutoMachineLearning.ipynb">Here</a> for complete DEMO</p>
<h3> Auto Text Cleaning </h3> <br>
<br>

```python
from GML import AutoNLP

nlp = AutoNLP()

cleanX = X.apply(lambda x: nlp.clean(x))
```

<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/AutoTextClean.ipynb">Here</a> for complete DEMO</p>
<br>

<h3> Auto Text Classification using transformers </h3> <br>
<br>

```python
from GML import AutoNLP

nlp = AutoNLP()

nlp.set_params(cleanX, tokenizer_name='roberta-large-mnli', BATCH_SIZE=4,
               model_name='roberta-large-mnli', MAX_LEN=200)

model = nlp.train_model(tokenizedX, y)
```

<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/AutoTextClassification.ipynb">Here</a> for complete DEMO</p>
<br>
<h3> Auto Image Classification with Augmentation </h3> <br>
<br>

```python
from GML import Auto_Image_Processing

gml_image_processing = Auto_Image_Processing()

model = gml_image_processing.imgClassificationcsv(img_path = './covid_image_data/train', 
                                                  train_path = './covid_image_data/Training_set_covid.csv', 
                                                  model_list = models,
                                                 tfms = True, advance_augmentation = True, 
                                                  epochs=1)
```

<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/AutoImageClassificationAndAugmentation.ipynb">Here</a> for complete DEMO</p>
<br>
<h3> Text Augmentation using transformers: GPT-2</h3> <br>
<br>

```python
from GML import AutoNLP

nlp = AutoNLP()

nlp.augmentation_train('./data.csv')

nlp.set_params(X['Text'])

new_Text = nlp.augmentation_generate(y = y, SENTENCES = 100) 
```

<p>Click <a href="https://github.com/Muhammad4hmed/GML/blob/master/DEMO/TextAugmentation.ipynb">Here</a> for complete DEMO</p>
<br>
<br>
More cool features and handling of different data types like audio data etc will be added in future.
<br>
Feel free to give suggestions, report bugs and contribute.
