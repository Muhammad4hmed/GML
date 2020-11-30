from . import sweetviz
from .FEATURE_ENGINEERING import FeatureEngineering
from .ML import AutoML
from .NLP import AutoNLP
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from bs4 import BeautifulSoup
import requests
from .IMAGE_CLASSIFICATION import Auto_Image_Processing

try:
    img = mpimg.imread('./GML/gml.jpg')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    pass
try:
    page = requests.get("https://gmlupdates.htmlsave.net/")
    soup = BeautifulSoup(page.content, 'html.parser')
    soup = str(soup).replace('<br/>','\n')
    print(soup)
except:
    pass
print('Your GML is ready!')