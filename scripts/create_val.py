import os
import json
import random
import shutil
import glob


def create_val_set(
    source_scene_dir,
    val_scene_dir,
    val_ratio=0.2,
    seed=42,
):
    """
    Create a validation split from a single BOP scene.

    The original scene is not modified. RGB, depth, masks and the
    corresponding entries in the BOP metadata files are copied to
    the validation scene.

    Expected source structure:

        scene/
        ├── rgb_realsense/
        ├── depth_realsense/
        ├── mask/
        ├── mask_visib/
        ├── scene_camera_realsense.json
        ├── scene_gt_realsense.json
        └── scene_gt_info_realsense.json
    """

    rng = random.Random(seed)

    os.makedirs(val_scene_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load BOP metadata.
    # ------------------------------------------------------------------

    def load_json(filename):
        path = os.path.join(source_scene_dir, filename)

        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            return json.load(f)

    scene_camera = load_json("scene_camera_realsense.json")
    scene_gt = load_json("scene_gt_realsense.json")
    scene_gt_info = load_json("scene_gt_info_realsense.json")

    if scene_camera is None:
        raise FileNotFoundError(
            f"Missing scene_camera_realsense.json in {source_scene_dir}"
        )

    # scene_camera is the authoritative list of image IDs.
    # image_ids = sorted(int(x) for x in scene_camera.keys())

    # if not image_ids:
    #     print(f"No images found in {source_scene_dir}")
    #     return []
    # ------------------------------------------------------------------
# Determine available image IDs from the actual RGB files.
# ------------------------------------------------------------------

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

    # Extract image IDs from filenames such as:
    #
    #     000002.png
    #     000017.png
    #
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

    # IDs present in both RGB and scene_camera.
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

    # ------------------------------------------------------------------
    # Select validation images.
    # ------------------------------------------------------------------

    num_val = max(1, int(len(image_ids) * val_ratio))

    val_image_ids = sorted(
        rng.sample(image_ids, num_val)
    )

    print(
        f"{os.path.basename(source_scene_dir)}: "
        f"{len(image_ids)} images -> "
        f"{len(val_image_ids)} validation images"
    )

    # ------------------------------------------------------------------
    # Copy RGB and depth.
    # ------------------------------------------------------------------

    image_directories = [
        ("rgb_realsense", "rgb_realsense"),
        ("depth_realsense", "depth_realsense"),
    ]

    for source_subdir, val_subdir in image_directories:

        source_dir = os.path.join(
            source_scene_dir,
            source_subdir,
        )

        val_dir = os.path.join(
            val_scene_dir,
            val_subdir,
        )

        if not os.path.isdir(source_dir):
            print(
                f"Warning: directory does not exist: {source_dir}"
            )
            continue

        os.makedirs(val_dir, exist_ok=True)

        for image_id in val_image_ids:

            # Standard BOP naming.
            filename = f"{image_id:06d}.png"

            source_path = os.path.join(
                source_dir,
                filename,
            )

            val_path = os.path.join(
                val_dir,
                filename,
            )

            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Image not found: {source_path}"
                )

            shutil.copy2(
                source_path,
                val_path,
            )

    # ------------------------------------------------------------------
    # Copy masks.
    #
    # BOP masks are normally named:
    #
    #     000001_000000.png
    #     000001_000001.png
    #     ...
    #
    # where the first number is the image ID and the second number is
    # the GT instance index.
    # ------------------------------------------------------------------

    mask_directories = [
        ("mask", "mask"),
        ("mask_visib", "mask_visib"),
    ]

    for source_subdir, val_subdir in mask_directories:

        source_dir = os.path.join(
            source_scene_dir,
            source_subdir,
        )

        val_dir = os.path.join(
            val_scene_dir,
            val_subdir,
        )

        if not os.path.isdir(source_dir):
            print(
                f"Warning: directory does not exist: {source_dir}"
            )
            continue

        os.makedirs(val_dir, exist_ok=True)

        for image_id in val_image_ids:

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

                filename = os.path.basename(source_path)

                val_path = os.path.join(
                    val_dir,
                    filename,
                )

                shutil.copy2(
                    source_path,
                    val_path,
                )

    # ------------------------------------------------------------------
    # Filter metadata by image ID.
    # ------------------------------------------------------------------

    def filter_by_image_id(data):
        if data is None:
            return None

        return {
            str(image_id): data[str(image_id)]
            for image_id in val_image_ids
            if str(image_id) in data
        }

    val_scene_camera = filter_by_image_id(
        scene_camera
    )

    val_scene_gt = filter_by_image_id(
        scene_gt
    )

    val_scene_gt_info = filter_by_image_id(
        scene_gt_info
    )

    # ------------------------------------------------------------------
    # Write metadata.
    # ------------------------------------------------------------------

    def save_json(filename, data):

        if data is None:
            return

        path = os.path.join(
            val_scene_dir,
            filename,
        )

        with open(path, "w") as f:
            json.dump(
                data,
                f,
                indent=2,
            )

    save_json(
        "scene_camera_realsense.json",
        val_scene_camera,
    )

    save_json(
        "scene_gt_realsense.json",
        val_scene_gt,
    )

    save_json(
        "scene_gt_info_realsense.json",
        val_scene_gt_info,
    )

    print(
        f"Created validation scene: "
        f"{val_scene_dir} "
        f"({len(val_image_ids)} images)"
    )

    return val_image_ids


def create_bop_validation_set(
    dataset_dir,
    val_dir,
    val_ratio=0.2,
    seed=42,
):
    """
    Create a validation split for an entire BOP dataset.
    """

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    os.makedirs(
        val_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Find scenes.
    # ------------------------------------------------------------------

    scene_names = sorted(
        name
        for name in os.listdir(dataset_dir)
        if os.path.isdir(
            os.path.join(dataset_dir, name)
        )
        and name.isdigit()
    )

    if not scene_names:
        raise RuntimeError(
            f"No BOP scenes found in {dataset_dir}"
        )

    print(
        f"Found {len(scene_names)} scenes."
    )

    all_selected_ids = {}

    # ------------------------------------------------------------------
    # Process every scene.
    # ------------------------------------------------------------------

    for scene_name in scene_names:

        source_scene_dir = os.path.join(
            dataset_dir,
            scene_name,
        )

        val_scene_dir = os.path.join(
            val_dir,
            scene_name,
        )

        selected_ids = create_val_set(
            source_scene_dir=source_scene_dir,
            val_scene_dir=val_scene_dir,
            val_ratio=val_ratio,
            seed=seed,
        )

        all_selected_ids[scene_name] = selected_ids

    # ------------------------------------------------------------------
    # Save a record of the split.
    # ------------------------------------------------------------------

    split_file = os.path.join(
        val_dir,
        "selected_image_ids.json",
    )

    with open(split_file, "w") as f:
        json.dump(
            all_selected_ids,
            f,
            indent=2,
        )

    total_images = sum(
        len(ids)
        for ids in all_selected_ids.values()
    )

    print()
    print("Validation set created.")
    print(f"Scenes: {len(scene_names)}")
    print(f"Images: {total_images}")
    print(f"Split file: {split_file}")


if __name__ == "__main__":

    DATASET_DIR = (
        "/home/datasets/fractal_dataset_v3_refined/train"
    )

    VAL_DIR = (
        "/home/datasets/fractal_dataset_v3_refined/val"
    )

    create_bop_validation_set(
        dataset_dir=DATASET_DIR,
        val_dir=VAL_DIR,
        val_ratio=0.2,
        seed=42,
    )
