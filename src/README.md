
# Source Code and Dependencies

## Dependencies
The following packages are required to run the code:

- `ase`
- `sklearn`
- `torch`
- `tqdm`
- `tensorboardX`

## Model Training
To train the model, use the following command:

```bash
python train.py --batch_size 128 --data_dir_train /data/cb_dataset/train.db
```

## Model Testing
To test the model, run:

```bash
python infer.py --data_dir /data/cb_dataset/test.db
```

## Testing on RRUFF Dataset
For testing the model on the RRUFF dataset, execute:

```bash
python inferRRUFF.py --data_dir /data/cb_dataset/RRUFF.db --mp_dir /data/cb_dataset/mpdata.db
```
