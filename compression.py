from datasets import load_dataset, Value, Features
import datasets
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
import queue
import threading
import subprocess
import os
from tqdm.auto import tqdm
from io import BytesIO
import logging
import hashlib
import tarfile
import json
from huggingface_hub import HfApi


class E621Dataset(IterableDataset):
    def __init__(self, date):
        # date in format e.g. 2024-07-02
        post_dataset_url = f"https://e621.net/db_export/posts-{date}.csv.gz"

        self.post_dataset = load_dataset("csv", data_files=post_dataset_url, na_filter=False)
        self.post_dataset = self.post_dataset["train"]

        # Make post dataset queryable
        self.post_id_to_idx_df = self.post_dataset.select_columns(["id"]).to_pandas()
        # Set post id as the index and keep the original index column
        # post id -> idx
        self.post_id_to_idx_df = self.post_id_to_idx_df.reset_index().set_index("id", verify_integrity=True)

        features = Features({
            '__key__': Value(dtype='string', id=None),
            '__url__': Value(dtype='string', id=None),
            'image': datasets.Image(decode=False, id=None)  # Decode image in this script to handle bad files
        })

        self.image_dataset = load_dataset("boxingscorpionbagel/e621-2024", streaming=True, features=features)
        self.image_dataset = self.image_dataset["train"]


    def _get_images(self):
        for index, row in enumerate(self.image_dataset):
            # Get post data here because pandas is not thread safe (cannot put it in compress function)
            post_id = int(row["__key__"])

            try:
                post_idx = self.post_id_to_idx_df.loc[post_id]
            except Exception as e:
                logging.getLogger(__name__).info(f"Encountered E621 post id {row['__key__']}, but it does not exist in post database")
                continue

            post_data = self.post_dataset[post_idx]

            yield { "row": row, "post_data": post_data }


    def __iter__(self):
        return iter(self._get_images())
    

SHARD_SIZE_LIMIT = 1_000_000_000  # 1 GB in bytes


def compress(job_queue, upload_queue, current_shard, stop_event):
    while not stop_event.is_set():
        try:
            post = job_queue.get(block=True, timeout=5)  # Wait 5 seconds
        except queue.Empty:
            continue

        row = post["row"]
        id = row['__key__']
        image_bytes = row["image"]["bytes"]

        post_data = post["post_data"]
        # Unwrap
        file_ext = post_data["file_ext"][0]
        rating = post_data["rating"][0]
        tag_string = post_data["tag_string"][0]
        md5 = post_data["md5"][0]

        # Only ignore file ext which are not images
        if file_ext not in { "jpg", "png", "gif" }:
            logging.getLogger(__name__).info(f"E621 post id {id} is not an image type. Post dataset say it is of type {file_ext}. Skipping")
            job_queue.task_done()
            continue

        # Filter out "rating:e young"
        if rating == "e":
            tags = set(tag_string.split())
            tag_blocklist = ["young"]
            aTagIsInBlocklist = bool(tags.intersection(set(tag_blocklist)))
            if aTagIsInBlocklist:
                logging.getLogger(__name__).info(f"E621 post id {id} is tagged rating:e young. Skipping")
                job_queue.task_done()
                continue

        # Validate file with e621 hash
        computed_hash = hashlib.md5(image_bytes).hexdigest()
        if computed_hash != md5:
            logging.getLogger(__name__).error(f"E621 post id {id} image did not match e621 hash! Computed hash: {computed_hash}, original hash {md5}. Skipping")
            job_queue.task_done()
            continue

        # Ensure the image can be loaded by PIL
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except Exception as e:
            logging.getLogger(__name__).info(f"E621 post id {id} image cannot be loaded with PIL. Got Exception: {e}. Skipping")
            job_queue.task_done()
            continue

        # Check image is actually of the format what e621 says it is
        if image.format == "PNG":
            mimetype = image.get_format_mimetype()
            if mimetype == "image/png":
                true_format = "png"
            elif mimetype == "image/apng":
                true_format = "apng"
            else:
                logging.getLogger(__name__).error(f"E621 post id {id} image was of format PNG, but has unexpected minetype {mimetype}. Skipping")
                job_queue.task_done()
                continue
        elif image.format == "JPEG":
            true_format = "jpg"
        elif image.format == "GIF":
            true_format = "gif"
        else:
            logging.getLogger(__name__).error(f"E621 post id {id} image has unexpected format {image.format}. Skipping")
            job_queue.task_done()
            continue

        # Log if e621 labelled the image with the wrong format
        if file_ext != true_format:
            logging.getLogger(__name__).info(f"E621 post id {id} image was labelled with the wrong format. Labelled with {file_ext}, but was actually {true_format}")

        # Write image to file
        src_path = f"{id}.{true_format}"
        with open(src_path, "wb") as f:
            f.write(image_bytes)

        # Convert to jpeg-xl if png or jpg
        if true_format in { "png", "jpg" }:
            dst_path = f"{id}.jxl"

            command = [
                "cjxl",
                src_path,
                dst_path,
                "--lossless_jpeg=1"
            ]

            try:
                subprocess.run(command, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                logging.getLogger(__name__).info(f"E621 post id {id} image of format {true_format} failed to convert to jpeg-xl. Error: {e}, stout: {e.stdout}, stderr: {e.stderr}")
                os.remove(src_path)
                job_queue.task_done()
                continue

            final_path = dst_path
        else:
            final_path = src_path

        # Save current state in across threads
        current_shard_for_file_format = current_shard.posts[true_format][-1]
        posts = current_shard_for_file_format["posts"]
        current_size = os.path.getsize(final_path)

        with current_shard.lock:
            posts.append(id)
            current_shard_for_file_format["total_size"] += current_size

            current_shard.progress_bar.update(1)

            if current_shard_for_file_format["total_size"] > SHARD_SIZE_LIMIT:
                current_shard.posts[true_format].append(
                    {
                        "posts": [],
                        "total_size": 0
                    }
                )
                upload_queue.put(true_format)

        job_queue.task_done()


SUPPORTED_FILE_FORMATS = { "png", "apng", "jpg", "gif" }


def upload(upload_queue, current_shard, stop_event):
    # Can only run on one thread
    shard_info_path = "shards.json"
    try:
        with open(shard_info_path, "r") as f:
            shard_info = json.load(f)
    except:
        shard_info = {
            file_format: {
                "last_shard_num": 0
            }
            for file_format in SUPPORTED_FILE_FORMATS
        }

    api = HfApi()

    while not stop_event.is_set():
        try:
            file_format = upload_queue.get(block=True, timeout=5)  # Wait 5 seconds
        except queue.Empty:
            continue

        with current_shard.lock:
            upload_shard = current_shard.posts[file_format].pop(0)

        shard_info[file_format]["last_shard_num"] += 1
        shard_num = shard_info[file_format]["last_shard_num"]

        logging.getLogger(__name__).info(f"E621 post id {upload_shard['posts'][0]} is the first post of shard {shard_num} for {file_format}")

        with open(shard_info_path, "w") as f:
            json.dump(shard_info, f)

        # Create archive
        archive_path = f"shard-{shard_num}.tar"
        with tarfile.open(archive_path, "w") as tar:
            for id in upload_shard["posts"]:
                src_path = f"{id}.{file_format}"
                if file_format in { "png", "jpg" }:
                    dst_path = f"{id}.jxl"
                    final_path = dst_path
                else:
                    final_path = src_path

                tar.add(final_path)

        # Upload archive
        logging.getLogger(__name__).info(f"Uploading archive {archive_path}")
        api.upload_file(
            path_or_fileobj=archive_path,
            path_in_repo=f"{file_format}/{archive_path}",
            repo_id="owu1/e621",
            repo_type="dataset",
        )
        logging.getLogger(__name__).info(f"Finished uploading archive {archive_path}")

        # Delete image files
        for id in upload_shard["posts"]:
            src_path = f"{id}.{file_format}"
            os.remove(src_path)

            if file_format in { "png", "jpg" }:
                dst_path = f"{id}.jxl"
                os.remove(dst_path)

        # Delete archive
        os.remove(archive_path)

        upload_queue.task_done()


def filter_post_dataset(row):
    if row["file_ext"] not in { "jpg", "png", "gif" }:
        return False

    # Filter out "rating:e young"
    if row["rating"] == "e":
        tags = set(row["tag_string"].split())
        tag_blocklist = ["young"]
        aTagIsInBlocklist = bool(tags.intersection(set(tag_blocklist)))
        return not aTagIsInBlocklist
    
    return True


class CurrentShard:
    def __init__(self, dataset):
        self.lock = threading.Lock()
        self.posts = {
            file_format: [
                # In a list, because once the total size is reached and the posts are archived and uploaded,
                # the next images are still being downloaded to disk. At index len() - 1 will be the current shard being created (images downloaded and proccessed to).
                # At index 0 will be the shard that is currently being uploaded.
                # e.g. Start with len(file_format) - 1 == 0. Download images and put into index 0. Once total_size is reached,
                # start archiving and uploading images from index 0. Simultaneously, file_format.append({ ... }), start saving images at index 1
                {
                    "posts": [],
                    "total_size": 0
                }
            ]
            for file_format in SUPPORTED_FILE_FORMATS
        }

        # Get post count estimation
        filtered_dataset = dataset.post_dataset.filter(
            filter_post_dataset,
            num_proc=8,
            desc="Filtering post dataset"
        )
        
        self.progress_bar = tqdm(
            range(0, len(filtered_dataset)),
            initial=0,
            desc="Steps"
        )


def main():
    num_proc = 16 * 2

    logging.basicConfig(filename="log.txt", level=logging.INFO)
    logging.getLogger(__name__).info("Starting...")

    dataset = E621Dataset("2024-07-18")
    current_shard = CurrentShard(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda post : post[0],
        num_workers=2,
        prefetch_factor=10
    )

    job_queue = queue.Queue(maxsize=num_proc)
    upload_queue = queue.Queue()
    stop_event = threading.Event()

    upload_thread = threading.Thread(target=upload, args=(upload_queue, current_shard, stop_event))
    upload_thread.start()
    compress_threads = [threading.Thread(target=compress, args=(job_queue, upload_queue, current_shard, stop_event)) for _ in range(num_proc)]
    for thread in compress_threads:
        thread.start()

    try:
        for step, post in enumerate(dataloader):
            job_queue.put(post)
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()

    logging.getLogger(__name__).info("Exiting. Waiting for threads to finish...")

    for thread in compress_threads:
        thread.join()

    upload_thread.join()


if __name__ == "__main__":
    main()
