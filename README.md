# Repair-QA


## Dataset


## Packages
To install required packages for this repository,
```
pip install -r requirements.txt
```

## Train the model
1. Download [Repair-QA]() dataset and place it under folder ```data```
2. Modify ```utils/config.py``` to point to the right folders.
3. Start training using the following command
    ```
    python engine.py
    ```

## Predict for a given input file
1. To make predictions for a input file.
    ```
    python predict.py <input_file> <prediction_savename>
    ```

Note: The input file should consists of a list of dictionaries with each item consisting of {"Context": list, "Question": str}

## Compute evaluation metrics on the prediction file
1. Compute metrics on the evaluation file
    ```
    python src/compute_metrics.py <prediction_file>
    ```

## Serve the model as a service

## Results
