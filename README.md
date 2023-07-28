# Unsupervised Lifelong Learning with Sustained Representation Fairness
Lifelong learning, pivotal in the incremental improvement of decision-making functions, can be ill-conditioned when dealing with one
or several upcoming tasks that insinuate spurious correlations between target labels and sensitive demographic attributes such as
gender, age, or ethnicity. This often results in biased decisions, disproportionately favoring certain demographic groups. Prior studies
to de-bias such lifelong learners by fostering fairness-aware, intermediate representations often overlook the inherent diversity of
task distributions, thereby faltering in ensuring fairness in a life-long fashion. This challenge intensifies in the context of unlabeled
tasks, where discerning distributional shifts for the adaptation of
learned fair representations is notably intricate. Motivated by this,
we propose Sustaining Fair R epresentations in Unsupervised Lifelong
Learning (FaRULi), a new paradigm inspired by human instinctive
learning behavior. Like human who tends to prioritize simpler tasks
over more challenging ones that significantly outstrip one’s current
knowledge scope, FaRULi strategically schedules a buffer of tasks
based on the proximity of their fair representations. The learner
starts from tasks that share similar fair representations, accumu-
lating essential de-biasing knowledge from them. Subsequently,
once the learner revisits a previously postponed task with more
divergent demographic distributions, it is more likely to increment
a fair representation from it, as the learner is now equipped with an
enriched knowledge base. FaRULi showcases promising capability
in making fair yet accurate decisions in a sequence of tasks without
supervision labels, backed by both theoretical results and empirical
evaluation on benchmark datasets.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.9.7.
* You have installed PyTorch 1.12.1.
## Running the Code

We share the code to run the Law School Dataset by the following command:
```bash
python train_alltasks_law.py
