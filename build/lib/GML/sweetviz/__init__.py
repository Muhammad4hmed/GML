# sweetviz public interface
# -----------------------------------------------------------------------------------
__title__ = 'sweetviz'
__version__ = "1.1.2"
__author__ = "Francois Bertrand"
__license__ = 'MIT'

# These are the main API functions
from .sv_public import analyze, compare, compare_intra
from .feature_config import FeatureConfig

# This is the main report class; holds the report data
# and is used to output the final report
from .dataframe_report import DataframeReport

# This is the config_parser, use to customize settings
from .config import config as config_parser
