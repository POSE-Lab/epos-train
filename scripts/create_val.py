import os
import json
import random
import shutil
import glob


def get_image_ids(scene_dir):
    """
    Get image IDs that exist in both RGB and scene_camera metadata.
    """

    rgb_dir = os.path.join(scene_dir, "rgb")

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(
            f"RGB directory not found: {rgb_dir}"
        )

    # --------------------------------------------------------------
    # IDs from RGB files
    # --------------------------------------------------------------

    rgb_image_ids = set()

    for filename in os.listdir(rgb_dir):

        if not filename.lower().endswith(".png"):
            continue

        stem = os.path.splitext(filename)[0]

        try:
            image_id = int(stem)
        except ValueError:
            print(
                f"Warning: ignoring unexpected RGB filename: "
                f"{filename}"
            )
            continue

        rgb_image_ids.add(image_id)

    # --------------------------------------------------------------
    # IDs from scene_camera
    # --------------------------------------------------------------

    camera_path = os.path.join(
        scene_dir,
        "scene_camera.json",
    )

    if not os.path.exists(camera_path):
        raise FileNotFoundError(
            f"Missing scene_camera.json: {camera_path}"
        )

    with open(camera_path, "r") as f:
        scene_camera = json.load(f)

    camera_image_ids = {
        int(image_id)
        for image_id in scene_camera.keys()
    }

    # --------------------------------------------------------------
    # Intersection
    # --------------------------------------------------------------

    image_ids = sorted(
        rgb_image_ids & camera_image_ids
    )

    if not image_ids:
        raise RuntimeError(
            f"No matching RGB/scene_camera image IDs "
            f"found in {scene_dir}"
        )

    # Warn about inconsistencies.

    rgb_only = rgb_image_ids - camera_image_ids
    camera_only = camera_image_ids - rgb_image_ids

    if rgb_only:
        print(
            f"Warning: {len(rgb_only)} RGB images have no "
            f"scene_camera entry."
        )

    if camera_only:
        print(
            f"Warning: {len(camera_only)} scene_camera entries "
            f"have no RGB image."
        )

    return image_ids


def load_scene_metadata(scene_dir):
    """
    Load all available BOP metadata files.
    """

    metadata = {}

    filenames = [
        "scene_camera.json",
        "scene_gt.json",
        "scene_gt_info.json",
    ]

    for filename in filenames:

        path = os.path.join(
            scene_dir,
            filename,
        )

        if os.path.exists(path):

            with open(path, "r") as f:
                metadata[filename] = json.load(f)

        else:

            metadata[filename] = None

    return metadata


def filter_metadata(data, image_ids):
    """
    Keep only entries corresponding to the selected image IDs.
    """

    if data is None:
        return None

    return {
        str(image_id): data[str(image_id)]
        for image_id in image_ids
        if str(image_id) in data
    }


def copy_scene_files(
    source_scene_dir,
    destination_scene_dir,
    image_ids,
):
    """
    Copy RGB, depth and masks for the selected image IDs.
    """

    os.makedirs(
        destination_scene_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # RGB and depth
    # --------------------------------------------------------------

    image_directories = [
        ("rgb", "rgb"),
        ("depth", "depth"),
    ]

    for source_subdir, destination_subdir in image_directories:

        source_dir = os.path.join(
            source_scene_dir,
            source_subdir,
        )

        destination_dir = os.path.join(
            destination_scene_dir,
            destination_subdir,
        )

        if not os.path.isdir(source_dir):

            print(
                f"Warning: directory does not exist: "
                f"{source_dir}"
            )

            continue

        os.makedirs(
            destination_dir,
            exist_ok=True,
        )

        for image_id in image_ids:

            filename = f"{image_id:06d}.png"

            source_path = os.path.join(
                source_dir,
                filename,
            )

            destination_path = os.path.join(
                destination_dir,
                filename,
            )

            if not os.path.exists(source_path):

                raise FileNotFoundError(
                    f"Image not found: {source_path}"
                )

            shutil.copy2(
                source_path,
                destination_path,
            )

    # --------------------------------------------------------------
    # Masks
    # --------------------------------------------------------------

    mask_directories = [
        ("mask", "mask"),
        ("mask_visib", "mask_visib"),
    ]

    for source_subdir, destination_subdir in mask_directories:

        source_dir = os.path.join(
            source_scene_dir,
            source_subdir,
        )

        destination_dir = os.path.join(
            destination_scene_dir,
            destination_subdir,
        )

        if not os.path.isdir(source_dir):

            print(
                f"Warning: directory does not exist: "
                f"{source_dir}"
            )

            continue

        os.makedirs(
            destination_dir,
            exist_ok=True,
        )

        for image_id in image_ids:

            prefix = f"{image_id:06d}_"

            pattern = os.path.join(
                source_dir,
                f"{prefix}*.png",
            )

            mask_files = glob.glob(pattern)

            for source_path in mask_files:

                filename = os.path.basename(
                    source_path
                )

                destination_path = os.path.join(
                    destination_dir,
                    filename,
                )

                shutil.copy2(
                    source_path,
                    destination_path,
                )


def write_scene_metadata(
    source_scene_dir,
    destination_scene_dir,
    image_ids,
):
    """
    Copy and filter BOP metadata for the selected image IDs.
    """

    metadata = load_scene_metadata(
        source_scene_dir
    )

    for filename, data in metadata.items():

        if data is None:
            continue

        filtered_data = filter_metadata(
            data,
            image_ids,
        )

        destination_path = os.path.join(
            destination_scene_dir,
            filename,
        )

        with open(destination_path, "w") as f:

            json.dump(
                filtered_data,
                f,
                indent=2,
            )


def create_scene_split(
    source_scene_dir,
    train_scene_dir,
    val_scene_dir,
    val_ratio=0.2,
    seed=42,
):
    """
    Create train/validation copies of one BOP scene.

    The source scene is never modified.

    Validation images are randomly selected according to
    val_ratio. All remaining images go into the training set.
    """

    # --------------------------------------------------------------
    # Get image IDs
    # --------------------------------------------------------------

    image_ids = get_image_ids(
        source_scene_dir
    )

    # --------------------------------------------------------------
    # Deterministic per-scene random selection
    # --------------------------------------------------------------

    rng = random.Random(seed)

    num_val = max(
        1,
        int(len(image_ids) * val_ratio),
    )

    # Avoid putting every image into validation.

    if num_val >= len(image_ids):
        num_val = len(image_ids) - 1

    val_image_ids = sorted(
        rng.sample(
            image_ids,
            num_val,
        )
    )

    val_image_ids_set = set(
        val_image_ids
    )

    train_image_ids = [
        image_id
        for image_id in image_ids
        if image_id not in val_image_ids_set
    ]

    train_image_ids = sorted(
        train_image_ids
    )

    # --------------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------------

    print()
    print(
        f"Scene: {os.path.basename(source_scene_dir)}"
    )

    print(
        f"  Original:    {len(image_ids)}"
    )

    print(
        f"  Train:       {len(train_image_ids)}"
    )

    print(
        f"  Validation:  {len(val_image_ids)}"
    )

    # --------------------------------------------------------------
    # Create train copy
    # --------------------------------------------------------------

    copy_scene_files(
        source_scene_dir=source_scene_dir,
        destination_scene_dir=train_scene_dir,
        image_ids=train_image_ids,
    )

    write_scene_metadata(
        source_scene_dir=source_scene_dir,
        destination_scene_dir=train_scene_dir,
        image_ids=train_image_ids,
    )

    # --------------------------------------------------------------
    # Create validation copy
    # --------------------------------------------------------------

    copy_scene_files(
        source_scene_dir=source_scene_dir,
        destination_scene_dir=val_scene_dir,
        image_ids=val_image_ids,
    )

    write_scene_metadata(
        source_scene_dir=source_scene_dir,
        destination_scene_dir=val_scene_dir,
        image_ids=val_image_ids,
    )

    return train_image_ids, val_image_ids


def create_bop_train_val_dataset(
    source_dataset_dir,
    output_dataset_dir,
    val_ratio=0.2,
    seed=42,
):
    """
    Create a complete BOP dataset containing:

        output_dataset_dir/
        ├── train/
        │   ├── 000000/
        │   ├── 000001/
        │   └── ...
        │
        ├── val/
        │   ├── 000000/
        │   ├── 000001/
        │   └── ...
        │
        └── split.json

    The original dataset is never modified.
    """

    if not os.path.isdir(source_dataset_dir):
        raise FileNotFoundError(
            source_dataset_dir
        )

    # --------------------------------------------------------------
    # Output directories
    # --------------------------------------------------------------

    train_dataset_dir = os.path.join(
        output_dataset_dir,
        "train",
    )

    val_dataset_dir = os.path.join(
        output_dataset_dir,
        "val",
    )

    os.makedirs(
        train_dataset_dir,
        exist_ok=True,
    )

    os.makedirs(
        val_dataset_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Find scenes
    # --------------------------------------------------------------

    scene_names = sorted(
        name
        for name in os.listdir(
            source_dataset_dir
        )
        if os.path.isdir(
            os.path.join(
                source_dataset_dir,
                name,
            )
        )
        and name.isdigit()
    )

    if not scene_names:

        raise RuntimeError(
            f"No BOP scenes found in "
            f"{source_dataset_dir}"
        )

    print(
        f"Found {len(scene_names)} scenes."
    )

    split_info = {}

    total_original = 0
    total_train = 0
    total_val = 0

    # --------------------------------------------------------------
    # Process scenes
    # --------------------------------------------------------------

    for scene_name in scene_names:

        source_scene_dir = os.path.join(
            source_dataset_dir,
            scene_name,
        )

        train_scene_dir = os.path.join(
            train_dataset_dir,
            scene_name,
        )

        val_scene_dir = os.path.join(
            val_dataset_dir,
            scene_name,
        )

        train_ids, val_ids = create_scene_split(
            source_scene_dir=source_scene_dir,
            train_scene_dir=train_scene_dir,
            val_scene_dir=val_scene_dir,
            val_ratio=val_ratio,
            seed=seed,
        )

        split_info[scene_name] = {
            "train": train_ids,
            "val": val_ids,
        }

        total_original += (
            len(train_ids) + len(val_ids)
        )

        total_train += len(train_ids)
        total_val += len(val_ids)

    # --------------------------------------------------------------
    # Save split information
    # --------------------------------------------------------------

    split_file = os.path.join(
        output_dataset_dir,
        "split.json",
    )

    with open(split_file, "w") as f:

        json.dump(
            split_info,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 60)
    print("BOP train/validation dataset created")
    print("=" * 60)

    print(
        f"Source:       {source_dataset_dir}"
    )

    print(
        f"Output:       {output_dataset_dir}"
    )

    print(
        f"Scenes:       {len(scene_names)}"
    )

    print(
        f"Original:     {total_original}"
    )

    print(
        f"Train:        {total_train}"
    )

    print(
        f"Validation:   {total_val}"
    )

    print(
        f"Split file:   {split_file}"
    )

    print("=" * 60)


if __name__ == "__main__":

    SOURCE_DATASET_DIR = (
        "/home/datasets/fractal_pbr_upright_single/train_pbr"
    )

    OUTPUT_DATASET_DIR = (
        "/home/datasets/fractal_pbr_upright_single_copy/"
    )

    create_bop_train_val_dataset(
        source_dataset_dir=SOURCE_DATASET_DIR,
        output_dataset_dir=OUTPUT_DATASET_DIR,
        val_ratio=0.3,
        seed=42,
    )
