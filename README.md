# Community Search (K-Truss) and Densest Subgraph Discovery
Repository of assignments from the part on "Mining on Large Graphs" of the University course "Advanced Topics in Computer Science"

### Introduction
This project focuses on analyzing and visualizing the Truss Decomposition and different k-trusses of a temporal network dataset. <br/>
Additionally, the project implements an efficient algorithm to find the densest subgraph and compares the results with the k-truss decomposition approach. <br/>
For each phase we produced polts and charts to better visualize how graphs are structured and better understand which are the temporal trends of the messages sent between users. <br/>

The dataset used is the CollegeMsg dataset, which includes 1,899 nodes and about 59,835 temporal edges (or 20,296 static edges) representing user communication data. <br/>
It's stored into "CollegeMsg.txt" where each row (a graph edge) consists of a source user (SRC), a target user (TGT), and a Unix timestamp (UNIXTS). <br/>
For more information about the dataset: (https://snap.stanford.edu/data/CollegeMsg.html) <br/>
<br/>
> [!NOTE]
> This project is developed in python 3.12 and use Pandas, NetworkX and MatPlotLib modules.
<br/>
### Project Structure

- **Dataset folder**: It stores the text file (`CollegeMsg.txt`) containing the communication data between users and a script to convert unix timestamps to datetime.
- **Graph plots folder**: It stores all the images about graphs' visual representation and metrics' charts.
- **Output log folder**: It stores text files where are written statistics and graphs' characteristics for each project phase (truss and dsg).
<br/>
- **graphScript.py**: This script performs truss decomposition and k-trusses of the input graph (the selected k values are [3, 4, 5, 6, 7]), then saves statistics and plot images for the resulting graphs.
- **dsg_greedy_plus_plus.py**: This script performs the greedy++ algorithm for dsg discovery and saves statistics and plot images for the resulting graphs.
- **dsg_discovery.py**: In addiction, we tried to add a second dsg discovery method in this script. The algorithm considered here uses a convex-programming approach, for more information: [A Convex-Programming Approach for Efficient Directed Densest Subgraph Discovery](https://dl.acm.org/doi/10.1145/3514221.3517837).
