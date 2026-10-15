import os
import json
import random
import shutil
import glob


def load_json(scene_dir, filename):
    path = os.path.join(scene_dir, filename)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def save_json(scene_dir, filename, data):
    if data is None:
        return

    path = os.path.join(scene_dir, filename)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_image_ids(source_scene_dir, scene_camera):
    """
    Get image IDs that exist both in RGB and scene_camera.
    """

    rgb_dir = os.path.join(
        source_scene_dir,
        "rgb_realsense",
    )

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(
            f"RGB directory not found: {rgb_dir}"
        )

    rgb_files = [
        filename
        for filename in os.listdir(rgb_dir)
        if filename.lower().endswith(".png")
    ]

    rgb_image_ids = set()

    for filename in rgb_files:

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

    camera_image_ids = {
        int(image_id)
        for image_id in scene_camera.keys()
    }

    image_ids = sorted(
        rgb_image_ids & camera_image_ids
    )

    if not image_ids:
        raise RuntimeError(
            f"No matching RGB/scene_camera image IDs found "
            f"in {source_scene_dir}"
        )

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


def copy_scene_subset(
    source_scene_dir,
    destination_scene_dir,
    image_ids,
):
    """
    Copy a subset of a BOP scene.

    Copies:
        rgb_realsense
        depth_realsense
        mask
        mask_visib

    and filters:
        scene_camera_realsense.json
        scene_gt_realsense.json
        scene_gt_info_realsense.json

    according to image_ids.
    """

    os.makedirs(
        destination_scene_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load metadata.
    # ------------------------------------------------------------------

    scene_camera = load_json(
        source_scene_dir,
        "scene_camera_realsense.json",
    )

    scene_gt = load_json(
        source_scene_dir,
        "scene_gt_realsense.json",
    )

    scene_gt_info = load_json(
        source_scene_dir,
        "scene_gt_info_realsense.json",
    )

    if scene_camera is None:
        raise FileNotFoundError(
            f"Missing scene_camera_realsense.json in "
            f"{source_scene_dir}"
        )

    # ------------------------------------------------------------------
    # Copy RGB and depth.
    # ------------------------------------------------------------------

    image_directories = [
        ("rgb_realsense", "rgb_realsense"),
        ("depth_realsense", "depth_realsense"),
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

    # ------------------------------------------------------------------
    # Copy masks.
    # ------------------------------------------------------------------

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

            if not mask_files:
                print(
                    f"Warning: no masks found for image "
                    f"{image_id} in {source_dir}"
                )

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

    # ------------------------------------------------------------------
    # Filter metadata.
    # ------------------------------------------------------------------

    def filter_by_image_id(data):

        if data is None:
            return None

        return {
            str(image_id): data[str(image_id)]
            for image_id in image_ids
            if str(image_id) in data
        }

    destination_scene_camera = filter_by_image_id(
        scene_camera
    )

    destination_scene_gt = filter_by_image_id(
        scene_gt
    )

    destination_scene_gt_info = filter_by_image_id(
        scene_gt_info
    )

    # ------------------------------------------------------------------
    # Save metadata.
    # ------------------------------------------------------------------

    save_json(
        destination_scene_dir,
        "scene_camera_realsense.json",
        destination_scene_camera,
    )

    save_json(
        destination_scene_dir,
        "scene_gt_realsense.json",
        destination_scene_gt,
    )

    save_json(
        destination_scene_dir,
        "scene_gt_info_realsense.json",
        destination_scene_gt_info,
    )


def get_scene_names(dataset_split_dir):
    """
    Find numeric BOP scene directories.
    """

    scene_names = sorted(
        name
        for name in os.listdir(dataset_split_dir)
        if os.path.isdir(
            os.path.join(dataset_split_dir, name)
        )
        and name.isdigit()
    )

    if not scene_names:
        raise RuntimeError(
            f"No BOP scenes found in {dataset_split_dir}"
        )

    return scene_names


def create_split_dataset(
    source_dataset_root,
    source_split="train",
    output_dataset_root=None,
    val_ratio=0.2,
    seed=42,
):
    """
    Create a completely separate copy of a BOP dataset with
    train/validation splits.

    Original dataset is NOT modified.

    Example:

        essential/
            train/
                000000/
                000001/

        becomes:

        essential2/
            train/
                000000/       <- 80%
                000001/       <- 80%

            val/
                000000/       <- 20%
                000001/       <- 20%
    """

    if output_dataset_root is None:
        output_dataset_root = (
            source_dataset_root.rstrip("/")
            + "_split"
        )

    source_split_dir = os.path.join(
        source_dataset_root,
        source_split,
    )

    output_train_dir = os.path.join(
        output_dataset_root,
        "train",
    )

    output_val_dir = os.path.join(
        output_dataset_root,
        "val",
    )

    if not os.path.isdir(source_split_dir):
        raise FileNotFoundError(
            f"Source split not found: {source_split_dir}"
        )

    # ------------------------------------------------------------------
    # Safety check.
    # ------------------------------------------------------------------

    if os.path.exists(output_dataset_root):
        raise FileExistsError(
            f"Output dataset already exists:\n"
            f"{output_dataset_root}\n\n"
            f"Delete it first if you want to recreate the split."
        )

    os.makedirs(
        output_train_dir,
        exist_ok=True,
    )

    os.makedirs(
        output_val_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Copy dataset-level files/directories.
    #
    # For example, if essential/ contains:
    #
    #   models/
    #   models_info.json
    #   train/
    #
    # then models/ and models_info.json are copied to essential2/.
    #
    # The original train/ directory is NOT copied because we create
    # our own split below.
    # ------------------------------------------------------------------

    print()
    print("Copying dataset-level files...")
    print()

    for name in os.listdir(source_dataset_root):

        if name == source_split:
            continue

        source_path = os.path.join(
            source_dataset_root,
            name,
        )

        destination_path = os.path.join(
            output_dataset_root,
            name,
        )

        if os.path.isdir(source_path):

            print(
                f"Copying directory: {name}"
            )

            shutil.copytree(
                source_path,
                destination_path,
            )

        else:

            print(
                f"Copying file: {name}"
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

    # ------------------------------------------------------------------
    # Find scenes.
    # ------------------------------------------------------------------

    scene_names = get_scene_names(
        source_split_dir
    )

    print()
    print(
        f"Found {len(scene_names)} scenes."
    )
    print()

    all_train_ids = {}
    all_val_ids = {}

    total_images = 0
    total_train = 0
    total_val = 0

    # ------------------------------------------------------------------
    # Split every scene.
    # ------------------------------------------------------------------

    for scene_index, scene_name in enumerate(scene_names):

        source_scene_dir = os.path.join(
            source_split_dir,
            scene_name,
        )

        train_scene_dir = os.path.join(
            output_train_dir,
            scene_name,
        )

        val_scene_dir = os.path.join(
            output_val_dir,
            scene_name,
        )

        # Use a different deterministic RNG per scene.
        #
        # This avoids every scene getting exactly the same random
        # sequence while still making the split reproducible.
        scene_rng = random.Random(
            seed + scene_index
        )

        # Load camera metadata.
        scene_camera = load_json(
            source_scene_dir,
            "scene_camera_realsense.json",
        )

        if scene_camera is None:
            raise FileNotFoundError(
                f"Missing scene_camera_realsense.json "
                f"in {source_scene_dir}"
            )

        image_ids = get_image_ids(
            source_scene_dir,
            scene_camera,
        )

        # --------------------------------------------------------------
        # Select validation images.
        # --------------------------------------------------------------

        num_val = max(
            1,
            int(len(image_ids) * val_ratio),
        )

        val_image_ids = sorted(
            scene_rng.sample(
                image_ids,
                num_val,
            )
        )

        val_image_ids_set = set(
            val_image_ids
        )

        train_image_ids = sorted(
            image_id
            for image_id in image_ids
            if image_id not in val_image_ids_set
        )

        print(
            f"Scene {scene_name}: "
            f"{len(image_ids)} total -> "
            f"{len(train_image_ids)} train + "
            f"{len(val_image_ids)} val"
        )

        # --------------------------------------------------------------
        # Copy train subset.
        # --------------------------------------------------------------

        copy_scene_subset(
            source_scene_dir=source_scene_dir,
            destination_scene_dir=train_scene_dir,
            image_ids=train_image_ids,
        )

        # --------------------------------------------------------------
        # Copy validation subset.
        # --------------------------------------------------------------

        copy_scene_subset(
            source_scene_dir=source_scene_dir,
            destination_scene_dir=val_scene_dir,
            image_ids=val_image_ids,
        )

        all_train_ids[scene_name] = train_image_ids
        all_val_ids[scene_name] = val_image_ids

        total_images += len(image_ids)
        total_train += len(train_image_ids)
        total_val += len(val_image_ids)

    # ------------------------------------------------------------------
    # Save split information.
    # ------------------------------------------------------------------

    split_info = {
        "source_dataset": source_dataset_root,
        "source_split": source_split,
        "val_ratio": val_ratio,
        "seed": seed,
        "total_images": total_images,
        "train_images": total_train,
        "val_images": total_val,
        "train": all_train_ids,
        "val": all_val_ids,
    }

    split_file = os.path.join(
        output_dataset_root,
        "selected_image_ids.json",
    )

    with open(split_file, "w") as f:
        json.dump(
            split_info,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("Dataset split created.")
    print("=" * 60)
    print(
        f"Source:       {source_dataset_root}"
    )
    print(
        f"Output:       {output_dataset_root}"
    )
    print(
        f"Scenes:       {len(scene_names)}"
    )
    print(
        f"Total images: {total_images}"
    )
    print(
        f"Train images: {total_train}"
    )
    print(
        f"Val images:   {total_val}"
    )
    print(
        f"Val ratio:    {val_ratio}"
    )
    print(
        f"Seed:         {seed}"
    )
    print(
        f"Split file:   {split_file}"
    )
    print("=" * 60)


if __name__ == "__main__":

    SOURCE_DATASET = (
        "/home/datasets/fractalminimal"
    )

    OUTPUT_DATASET = (
        "/home/datasets/fractalminimal_val"
    )

    create_split_dataset(
        source_dataset_root=SOURCE_DATASET,
        source_split="train_pbr",
        output_dataset_root=OUTPUT_DATASET,
        val_ratio=0.1,
        seed=42,
    )
