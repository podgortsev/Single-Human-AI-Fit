# shared

Datasets used by more than one experiment, each next to the script that builds
it. Builders are deterministic (fixed seed); the committed `.json` / `.csv` are
what the experiments actually used.

    profiles/   build_profiles.py    100 candidate descriptions
                profiles.json        used by methods 1, 4, 6
                profiles.csv         same data, flat

    tasks/      build_tasks.py       200 arithmetic tasks with keys, method 3
                tasks.json           200 rows over 191 distinct questions
                tasks.csv            same data, flat
                validate_tasks.py    the independent solver: re-derives every
                                     key from the question text. All 200 match.

    questions/  build_questions.py   60 open-ended questions, no keys
                questions.json       used by methods 2 and 7
                questions.csv        same data, flat

    axes/       build_axes.py        28 axes of deviation, by type not category
                axes.json            id, group, signal type, turn count, example
                axes.csv             same data, flat
                axes_design.md       why organised by type of deviation

`axes.json` is a design catalogue. No runner reads it; the condition strings
actually sent to the models live in each experiment's run script.

Three known properties of `tasks.json`, all reported by `validate_tasks.py` and
none of them changing a published number: 9 of the 200 rows repeat a question
that already appears (with consistent keys), the date family is ambiguous
between an inclusive and an exclusive reading of "runs for N days", and the
table generator does not exclude a tie (the shipped set contains none). The
dataset is deliberately not regenerated: every method 3 accuracy number was
measured on these exact rows.

Experiment scripts open these by bare filename (Colab upload). To run or
validate a script offline, copy the dataset it needs next to it first.
