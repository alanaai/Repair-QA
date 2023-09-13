# Repair-QA: No That's Not What I Meant: Handling Third Position Repair in Conversational Question Answering

![Alana AI](https://alanaai.com/wp-content/uploads/2020/08/ALANA_PURPLE.png)

![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)

This repo contains the dataset and implementation for the paper: <strong>No That's Not What I Meant: Handling Third Position Repair in Conversational Question Answering.</strong> [Vevake Balaraman](https://scholar.google.com/citations?user=GTtAXeIAAAAJ), [Arash Eshghi](https://scholar.google.com/citations?user=yCku-o8AAAAJ), [Ioannis Konstas](https://scholar.google.com/citations?user=FAJSqSkjAoIC) and [Ioannis Papaioannou](https://scholar.google.com/citations?user=gC0w0PIAAAAJ). [<strong>SIGdial 2023</strong>](https://sigdialinlg2023.github.io/index.html). [PDF](https://sigdialinlg2023.github.io/static/papers/sigdial/59_Paper.pdf)

## Dataset
The dataset can be found ```data``` folder


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
