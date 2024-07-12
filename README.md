```markdown

(For the facility location problem, we tested a flip-based local search and a recombination method with Markov chains. Chicago transportation data is used for calculating travel times between any two blocks in the city. We use a shortest path algorithm to estimate the travel times between any pair of blocks. We also implemented a reinforcement learning algorithm to improve the result we obtained from the recombination method. Once I submit the research paper, I will clean the code and explain the components.)

# Allocation of Primary Care Centers in the Chicago Area 

This project aims to optimize the allocation of primary care centers in Chicago using advanced data analysis and optimization techniques.

## Prerequisites

Before you begin, ensure you have Python installed on your system. This project is developed using Python 3.12. While it is likely that earlier versions of Python 3 may also work, they have not been tested and thus cannot be guaranteed to be fully compatible. For best results, using Python 3.12 or higher is recommended.

## Installation

Pipenv is used for managing project dependencies and virtual environments. Follow these steps to set up the project environment using Pipenv:

```bash
# Install Pipenv
pip install pipenv

# Clone the Project Repository
git clone https://github.com/kirtisoglu/Allocation-of-Primary-Care-Centers-in-Chicago
cd Allocation-of-Primary-Care-Centers-in-Chicago

#Set Up the Environment
pipenv install

# Activate the Virtual Environment
pipenv shell

#Deactivate the Environment:
exit
```

## DATA

This project utilizes several datasets to analyze and optimize the allocation of resources in Chicago Area. Data is not stored within the project repository due to its size. Ensure you follow data setup instructions in [data.txt](data.txt) to properly configure your environment for data handling.

## Running the Project

To run the project scripts, ensure you are within the activated Pipenv environment and execute Python scripts as follows:

```bash
python file_script.py
```

Replace `file_script.py` with the actual script file you want to execute.

## Contributing

Contributors are welcome! Please fork the repository and submit pull requests with your proposed changes. Ensure you follow the project's code style and guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.









