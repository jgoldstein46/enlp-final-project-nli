# enlp-final-project-nli


## run_experiments.py
Run models

`python run_experiments.py [options]`

Options:

|abbrev| command| description|
| ---- |----| -----|
| -b | --batch_size | batch size before weights update,default:32 |
| -l | --n_layers | number of layers, default:2 |
| -do | --dropout | dropout rate, default:0.5 |
| -em | --embedding-size | Embedding dimension size, default:300 |
| -hs | --hidden-size | Hidden layer size, default:300 |
| -d | --device | Either "cpu" or "cuda", default: "cpu" |
| -e | --epochs | Number of epochs, default:20 |
| -s | --seed | Train, dev, test split random seed number, default:100 |
| -t | --train | file to be used for training and dev and testing after split, default: "train.csv" |
| -c | --classifier | The name of the classifier you wish to use. baseline --> approach one, bert --> approach two, lstm --> approach three, gru --> approach four, default: "baseline" |
| -g | --glove-file | path to glove embeddings file, default: "../../data/glove.840B.300d.txt" |
