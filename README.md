# Hierarchical Facility Location with a Case Study of Chicago Healthcare Network

This project tackles the complex challenge of hierarchical facility location optimization, considering all possible design conditions. Our approach combines advanced mathematical modeling with innovative computational methods to precisely solve the problem on a large-scale dataset.

Key Features:
Integer Programming Formulation: We've developed a robust mathematical model using integer programming to represent the facility location problem.
Markov Chain Optimization: Our solution employs an iterative Markov chain process, where each state represents a valid graph partitioning corresponding to a geographical redistricting scheme.
A Novel Sampling Method: Inspired by the ReComb method (DeFord et al., 2019), we have created an efficient spanning tree sampling technique to explore the solution space efficiently.
Local Search Algorithms: We have improved upon two flip-based local search algorithms, providing a benchmark for our sampling method.
Case Study: Chicago Healthcare Accessibility
Our research applies this methodology to a real-world scenario: optimizing the location of new primary care facilities in Chicago to reduce healthcare access inequality via public transportation.
Transportation Network Analysis: We have constructed a real-world transportation network and calculated travel times between candidate facility locations and Census block centroids.
Results: Our approach yields convincing results in a short timeframe.
Current Status: We are in the final stages of preparing our research for publication. Stay tuned for updates on our submission and findings.
This project provides a practical implementation method for real-world scenarios of the facility location problem, with potential applications across various sectors including healthcare, retail, and public services.

Data
? Chicago Transit Authority GTFS data
? Cook County street data
2020 Illinois Census data at the block level
2020 Chicago tract IDs
? Chicago primary care centers, library, and school datasets 

Note: Illinois block data is removed due to its size. You can find it [here]{https://redistrictingdatahub.org/state/illinois}.
We extract its Chicago sub-data by matching it with 2020 Chicago tract IDs provided by the City of Chicago [here](https://www.chicago.gov/content/dam/city/depts/fin/municipal_depository/Addendum_2_Attachment_A_Chicago_Census_Tract%20_11_digit.pdf). We cleaned the data and saved it as chicago_tracts.csv in our data folder.



## Prerequisites

This project is developed using Python 3.12. While it is likely that earlier versions of Python 3 may also work, they have not been
tested and thus cannot be guaranteed to be fully compatible. For best results, using Python 3.12 or higher is recommended.

## Installation

Pipenv is used for managing project dependencies and virtual environments. Follow these steps to set up
the project environment using Pipenv:

```markdown
# Install Pipenv
pip install pipenv

# Clone the Project Repository
git clone https://github.com/kirtisoglu/Allocation-of-Primary-Care-Centers-in-Chicago
cd Allocation-of-Primary-Care-Centers-in-Chicago

# Set Up the Environment
pipenv install

# Activate the Virtual Environment
pipenv shell

#Deactivate the Environment:
exit
```

## DATA

This project utilizes several datasets to analyze and optimize the allocation of resources in the Chicago Area. 
Data is not stored within the project repository due to its size. Ensure you follow data setup instructions in 
data.txt](data.txt) to properly configure your environment for data handling.

## Running the Project

To run the project scripts, ensure you are within the activated Pipenv environment and execute Python scripts as follows:

```bash
python file_script.py
```

Replace `file_script.py` with the actual script file you want to execute.

## Contributing

Contributors are welcome! Please fork the repository and submit pull requests with your proposed changes. 
Ensure you follow the project's code style and guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.






