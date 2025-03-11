# Circuit Board Layout Optimizer 
This project uses a **Neural Network + FastAPI** to optimize the placement of circuit board components.

##  How to Run:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `python main.py`
3. Send a POST request to `http://localhost:8002/predict` with board input.

## Example API Input:
```json
{
    "board_input": [8,2,3,7,6,1,9,5,4,10]
}
