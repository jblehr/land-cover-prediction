# Land Cover Classification from Satellite Imagery

Our project attempts to use deep neural network models to model both spatial and temporal dimensions in satellite imagery. We specifically attempted to predict future land cover classification based on past satellite imagery, employing the new Dynamic EarthNet dataset. Facing data management challenges in the size and lack of diversity of the satellite imagery data, we focused on present-time land cover classification, while still testing temporal model performance. Our modeling approach included present-time land cover classification using a baseline logistic regression model to classify each pixel, then extended to a convolutional architecture (CNN) for next-timestep predictions, and finally, to a temporal CNN-GRU (Gated Recurrent Unit), or ConvGRU model for predictions leveraging past images. 

## Data structure

## Hyperparameter Optimization

First, ensure that data are structured properly as described in _Data Structure_ above. 

In order to optimize hyperparameters with the optuna package, run:

```
python3 scripts/convGRU.py --optuna --optuna_path sqlite:////path/to/desired/optuna.db
```

As currently set up, this will optimize:

- Number of hidden Layers in convGRU
- Number of hidden channels in each layer (can be different per layer)
- Convolutional kernel size between layers (fixed at one size per trial)

To train the final version of the model, simply omit the optuna flags:

```
python3 scripts/convGRU.py
```
