# Community Search (K-Truss) and Finding Densest Subgraph
Repository of assignments from the part on Mining on Large Graphs of the University course "Advanced Topics in Computer Science"

This project focuses on analyzing and visualizing the Truss Decomposition and different k-trusses of a temporal network dataset. <br/>
Additionally, the project implements an efficient algorithm to find the densest subgraph and compares the results with the k-truss decomposition approach. <br/>

The dataset used is the CollegeMsg dataset, which includes 1899 nodes and about 60,000 edges representing user communication data. <br/>
It's stored into "CollegeMsg.txt" where each row (graph edge) consists of a source user (SRC), a target user (TGT), and a Unix timestamp (UNIXTS). <br/>
For more information about the dataset: (https://snap.stanford.edu/data/CollegeMsg.html) <br/>
<br/>
> [!NOTE]
> This project is developed in python 3.12 and use Pandas, NetworkX and MatPlotLib modules.

### Project Structure

- **Dataset**: A text file (`CollegeMsg.txt`) containing the communication data between users.
- **Python Scripts**: The first (`dataScript.py`) converts unix timestamps of edges into date time; The second (`graphScript.py`) performs truss decomposition and k-trusses, then create plot images for the resulting graphs; The last (`dsg.py`) performs the Densest Subgraph discovery algorithm comparing the results.
- **Analytics**: A text file where are stored all the dataset characteristics and algorithms results.
- **Graphs Visualization**: Additional files for visualizing the graphs.
