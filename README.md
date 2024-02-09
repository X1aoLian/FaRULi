# Metric-Agnostic Continual Learning for Sustainable Group Fairness
Data-driven systems are increasingly integral to organizational
decision-making processes, e.g., hiring [2 ]. However, such systems
tend to establish spurious correlation between protected user char-
acteristics and prediction labels, resulting in decisions that are
discriminatory against certain demographic groups. Whereas prior
methods have focused on learning intermediate representations
that debias these characteristics to ensure fairness, they struggle
in continual learning (CL) contexts, where protected feature distri-
butions often shift in sequence of new tasks. How to learn inter-
mediate, debiased representations that can adapt to a sequence of
diverse tasks is a challenging and ongoing problem. In this paper,
we explore an even more challenging yet sustainable setting – a
system that learns from only one labeled task is expected to make
fair decisions across all subsequent unlabeled tasks. To solve it, we
propose MacFRL, a novel CL algorithm that fosters gradual and
structured unsupervised domain adaptation (UDA) to ensure that
group fairness is sustained across all tasks in dynamic environ-
ments. Our key idea is that, the more similar the protected feature
distributions of two tasks, the more likely that the decision function
learned from one task can adapt to the other through task-invariant,
debiased intermediate representation. Thus, MacFRL reorders the
sequence of tasks by their distance to the labeled task, performs
UDA on more similar tasks and gleans knowledge from them, and
then progressively adapts to those initially more distant tasks. The-
oretical results rationale our MacFRL solution. Experimental studies
substantiate that MacFRL outperforms its CL competitors in terms
of prediction accuracy, demographic parity, and equalized odds.
## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.9.7.
* You have installed PyTorch 1.12.1.
## Running the Code

We share the code to run the Law School Dataset by the following command:
```bash
python train_alltasks_law.py
