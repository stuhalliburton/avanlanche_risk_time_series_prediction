# Avalanche Risk Time Series Prediction

## Datasets

Download the SAIS snow profile datasets from here: https://www.sais.gov.uk/snow-profiles

## CSV Pre Formatting

The CSVs from SAIS come strangely formatted requiring some pre-formatting

Copy the CSV to `./profiles`

Open file in Vim and remove '=' signs.

```bash
%s/=//g
```

## Usage

Train the network using time series prediction.

```bash
python main.py
```
