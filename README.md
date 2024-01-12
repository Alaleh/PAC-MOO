This code is tested using `python 3.8`

Installing the dependencies:

`pip install -r requirements.txt`

Example of a command to run the code:

`python run_pacmoo.py --problem_name OSY --initial_points 6 --iterations 100 --preferences 0.8 0.2 --balance 0.5 0.5 --seeds 5`

The problem name, number of initial points, number of BO iterations, preferences for objectives, the balance between objectives and constraints, and the number of seeds can be changed.

If you use this code please cite our paper:

```bibtex

  @inproceedings{10.1145/3632410.3632427, 
  author = {Ahmadianshalchi, Alaleh and Belakaria, Syrine and Doppa, Janardhan Rao}, 
  title = {Preference-Aware Constrained Multi-Objective Bayesian Optimization}, 
  year = {2024}, 
  isbn = {9798400716348}, 
  publisher = {Association for Computing Machinery}, 
  address = {New York, NY, USA}, url = {https://doi.org/10.1145/3632410.3632427}, 
  doi = {10.1145/3632410.3632427}, 
  booktitle = {Proceedings of the 7th Joint International Conference on Data Science \& Management of Data (11th ACM IKDD CODS and 29th COMAD)}, 
  pages = {182–191}, 
  numpages = {10}, 
  location = {, Bangalore, India, }, 
  series = {CODS-COMAD '24} }

````
