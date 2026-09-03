# shared

Datasets used by more than one experiment, each next to the script that builds
it. Builders are deterministic (fixed seed); the committed `.json` / `.csv` are
what the experiments actually used.

    profiles/   build_profiles.py    100 candidate descriptions
                profiles.json        used by methods 1, 4, 6
                profiles.csv         same data, flat

    tasks/      build_tasks.py       200 arithmetic tasks, keys verified by an
                tasks.json           independent solver, used by method 3
                tasks.csv            same data, flat

    axes/       build_axes.py        28 axes of deviation, by type not category
                axes.json            id, group, signal type, turn count, example
                axes.csv             same data, flat
                axes_design.md       why organised by type of deviation

Experiment scripts open these by bare filename (Colab upload). To run or
validate a script offline, copy the dataset it needs next to it first.
