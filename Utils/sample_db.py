from osprey.config import Config
from sys import argv
from osprey.trials import make_session, Trial

if len(argv) != 3:
    print('usage: sample_db.py config.yaml sample_size')

inp_file = argv[1]
num = argv[2]

# Get original database and history
config1 = Config(inp_file)
df1 = config1.trial_results()
hist1 = config1.trials().query(Trial).all()


# Main loop
for name, group in df1.groupby('project_name'):
    # Sample the group
    sample = group.sample(int(num))
    keep = sample['id'].values

    db2 = make_session('sqlite:///osprey-trials-{}.db'.format(num), project_name=name)

    # Get the relevant trial objects from original db
    save_trials = [t for t in hist1 if t.project_name == name and t.id in keep and t.status == 'SUCCEEDED']
    print('Project name {0} \n Adding {1} trials'.format(name, len(save_trials)))

    # merge them into new db
    for trial in save_trials:
        db2.merge(trial)
    db2.commit()
    db2.close()




