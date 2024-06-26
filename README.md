### Add a new dataset

To add a dataset to the hub, you need to login using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Also install the ffmpeg encoder with the `libx264` codecs. Do not install it with `apt install`, try `conda install -c conda-forge ffmpeg`.

Then point to your raw dataset folder (e.g. `data/aloha_static_pingpong_test_raw`), and push your dataset to the hub with:
```bash
python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir /path/to/original/dataset
--raw-format <format name>
--repo-id <use_id/user_name>
--local-dir </a/path/user_id/user_name>
--push-to-hub <0 if you dont want upload the dataset, 1 otherwise>
--force-override <0 if you want to override previous converted dataset, 1 otherwise>
--num-workers <choose a smaller number if the computer crashes>
```

See `python lerobot/scripts/push_dataset_to_hub.py --help` for more instructions.

If your dataset format is not supported, implement your own in `lerobot/common/datasets/push_dataset_to_hub/${raw_format}_format.py` by copying examples like [pusht_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py), [umi_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/umi_zarr_format.py), [aloha_hdf5](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py), or [xarm_pkl](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/xarm_pkl_format.py). And then add the format name in `push_dataset_to_hub.py` [here](https://github.com/huggingface/lerobot/blob/342f429f1c321a2b4501c3007b1dacba7244b469/lerobot/scripts/push_dataset_to_hub.py#L61).

### An example that transfers our LMDB dataset to LeRobot format

First, install lmdb with `pip install lmdb`.

Then download the lmdb dataset of Calvin. I use HF-Mirror to download it. You can set the environment variable `export HF_ENDPOINT=https://hf-mirror.com` to avoid the connection problem in some regions.

```
apt install git-lfs aria2 curl
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh StarCycle/calvin_lmdb --dataset --tool aria2c -x 9
```

Now move the modified `lmdb_format.py` and `push_dataset_to_hub.py` to specific locations, and run:

```
python lerobot/scripts/push_dataset_to_hub.py  --raw-dir path/to/lmdb/folder --raw-format lmdb --repo-id StarCycle/test --local-dir StarCycle/test --push-to-hub 0 --force-override 1
```
