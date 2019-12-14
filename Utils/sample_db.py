from osprey.config import Config
from sys import argv, exit
from osprey.trials import make_session, Trial
from sklearn.model_selection import KFold


if len(argv) != 4:
    print('usage: sample_db.py config.yaml sample_size n_samples')
    exit(1)

inp_file = argv[1]
num = int(argv[2]) # TOTAL size of samples to use e.g. 100
iter = int(argv[3])  # Number of splits e.g. 5
# This will give 5 splits of 20 samples each.100

if num % iter != 0:
    print('sample_size not strictly divisible by n_samples')
    exit(1)

# Get original database and history
config1 = Config(inp_file)
df1 = config1.trial_results()
hist1 = config1.trials().query(Trial).all()


# Main loop
for name, group in df1.groupby('project_name'):
    # Sample the group
    sample = group.sample(num, random_state=42)
    cv = KFold(n_splits=iter, random_state=42)
    all_keep = sample['id'].values
    for i, (_, test_idx) in enumerate(cv.split(all_keep)):

        keep = all_keep[test_idx]

        db2 = make_session('sqlite:///osprey-trials-{0}-{1}.db'.format(int(num/iter), i), project_name=name)

        # Get the relevant trial objects from original db
        save_trials = [t for t in hist1 if t.project_name == name and t.id in keep and t.status == 'SUCCEEDED']
        print('Project name {0} Adding {1} trials {2}'.format(name, len(save_trials), keep))

        # merge them into new db
        for trial in save_trials:
            db2.merge(trial)
        db2.commit()
        db2.close()




