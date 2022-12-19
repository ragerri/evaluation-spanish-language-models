# Evaluation of Spanish Language Models

To reproduce results reported in the paper ["Lessons learned from the evaluation of Spanish Language Models"](https://arxiv.org/abs/2212.08390) paper:  

1. Generate the shell scripts by executing one of the *generate_scripts* contained in each task directory (you may want to play with the specified hyperparameters).
2. Run the generated shell scripts from the task directory.
3. The results of the experiments will be saved in the outputs directory.

## Example

1. cd cometa
2. python3 generate_scripts.py
3. (using bash): for i in *.sh; do ./$i; done

### Contact

Rodrigo Agerri<br>
rodrigo.agerri@ehu.eus<br>
HiTZ Center - Ixa, University of the Basque Country UPV/EHU
