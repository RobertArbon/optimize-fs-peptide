# Imports
from osprey.config import Config

# Load Configuation File
my_config = 'config.yaml'
config = Config(my_config)

# Retrieve Trial Results
df = config.trial_results()

