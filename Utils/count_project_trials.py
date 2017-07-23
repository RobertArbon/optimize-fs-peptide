# Imports
from osprey.config import Config
import sys

if len(sys.argv) != 2:
    print('Usage: count_project_trails [config file]')

# Load Configuation File
my_config = sys.argv[1] 

config = Config(my_config)

# Retrieve Trial Results
df = config.trial_results()

print(df['project_name'].value_counts())
