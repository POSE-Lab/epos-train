import os
import time
import pickle

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import tf2onnx
import onnxruntime as ort

from epos_lib import model
from epos_lib import common
from epos_lib import misc
from epos_lib import datagen
from epos_lib import corresp
from epos_lib import config

import pyprogressivex


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = "/home/store/tf_models/essential2/train"
OUTPUT_DIR = "/home/store/tf_models/essential2/frozen"

CHECKPOINT = None
# Example:
# CHECKPOINT = "/home/store/tf_models/essential2/train/model.ckpt-1000000"
#
# If None, the latest checkpoint in MODEL_DIR is used.

PB_PATH = os.path.join(
    OUTPUT_DIR,
    "essential2_frozen.pb"
)

ONNX_PATH = os.path.join(
    OUTPUT_DIR,
    "essential2.onnx"
)

IMAGE_PATH = (
    "/home/datasets/essential2/val/000000/"
    "rgb_realsense/000001.png"
)

NUM_OBJECTS = 1

# EPOS-specific configuration.
# These values should match the params used to train the checkpoint.
PARAMS_PATH = os.path.join(
    "/home/store/tf_models/essential2/",
    common.PARAMS_FILENAME
)

# Camera intrinsics.
# These are NOT image dimensions.
FX = 640.9191284179688
FY = 639.4614868164062
CX = 631.1490478515625
CY = 363.1187744140625

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'master', '',
    'BNS name of the tensorflow server')
flags.DEFINE_boolean(
    'cpu_only', False,
    'Whether to run the inference on CPU only.')
flags.DEFINE_string(
    'task_type', common.LOCALIZATION,  # LOCALIZATION, DETECTION
    'Type of the 6D object pose estimation task.')
flags.DEFINE_list(
    'infer_tfrecord_names', None,
    'Names of tfrecord files (without suffix) used for inference.')
flags.DEFINE_integer(
    'infer_max_height_before_crop', '1280',
    'Maximum image height before cropping (the image is downscaled if larger).')
flags.DEFINE_list(
    'infer_crop_size', '720,1280',
    'Image size [height, width] during inference.')
flags.DEFINE_string(
    'checkpoint_name', None,
    'Name of the checkpoint to evaluate (e.g. "model.ckpt-1000000"). The latest '
    'available checkpoint is used if None.')
flags.DEFINE_boolean(
    'project_to_surface', False,
    'Whether to project the predicted 3D locations to the object model.')
flags.DEFINE_boolean(
    'save_estimates', True,
    'Whether to save pose estimates in format expected by the BOP Challenge.')
flags.DEFINE_boolean(
    'save_corresp', False,
    'Whether to save established correspondences to text files.')
flags.DEFINE_string(
    'infer_name', None,
    'Name of the inference used in the filename of the saved estimates.')

# Pose fitting parameters.
flags.DEFINE_string(
    'fitting_method', common.PROGRESSIVE_X,  # PROGRESSIVE_X, OPENCV_RANSAC
    'Pose fitting method.')
flags.DEFINE_float(
    'inlier_thresh', 4.0,
    'Tau_r in the CVPR 2020 paper. Inlier threshold [px] on the '
    'reprojection error.')
flags.DEFINE_float(
    'neighbour_max_dist', 20.0,
    'Tau_d in the CVPR 2020 paper.')
flags.DEFINE_float(
    'min_hypothesis_quality', 0.5,
    'Tau_q in the CVPR 2020 paper')
flags.DEFINE_float(
    'required_progx_confidence', 0.5,
    'The required confidence used to calculate the number of Prog-X iterations.')
flags.DEFINE_float(
    'required_ransac_confidence', 1.0,
    'The required confidence used to calculate the number of RANSAC iterations.')
flags.DEFINE_float(
    'min_triangle_area', 0.0,
    'Tau_t in the CVPR 2020 paper.')
flags.DEFINE_boolean(
    'use_prosac', False,
    'Whether to use the PROSAC sampler.')
flags.DEFINE_integer(
    'max_model_number_for_pearl', 5,
    'Maximum number of instances to optimize by PEARL. PEARL is turned off if '
    'there are more instances to find.')
flags.DEFINE_float(
    'spatial_coherence_weight', 0.1,
    'Weight of the spatial coherence in Graph-Cut RANSAC.')
flags.DEFINE_float(
    'scaling_from_millimeters', 0.1,
    'Scaling factor of 3D coordinates when constructing the neighborhood graph. '
    '0.1 will convert mm to cm. See the CVPR 2020 paper for details.')
flags.DEFINE_float(
    'max_tanimoto_similarity', 0.9,
    'See the Progressive-X paper.')
flags.DEFINE_integer(
    'max_correspondences', None,
    'Maximum number of correspondences to use for fitting. Not applied if None.')
flags.DEFINE_integer(
    'max_instances_to_fit', None,
    'Maximum number of instances to fit. Not applied if None.')
flags.DEFINE_integer(
    'max_fitting_iterations', 400,
    'The maximum number of fitting iterations.')

# Visualization parameters.
flags.DEFINE_boolean(
    'vis', False,
    'Global switch for visualizations.')
flags.DEFINE_boolean(
    'vis_gt_poses', False,
    'Whether to visualize the GT poses.')
flags.DEFINE_boolean(
    'vis_pred_poses', True,
    'Whether to visualize the predicted poses.')
flags.DEFINE_boolean(
    'vis_gt_obj_labels', False,
    'Whether to visualize the GT object labels.')
flags.DEFINE_boolean(
    'vis_pred_obj_labels', True,
    'Whether to visualize the predicted object labels.')
flags.DEFINE_boolean(
    'vis_pred_obj_confs', False,
    'Whether to visualize the predicted object confidences.')
flags.DEFINE_boolean(
    'vis_gt_frag_fields', False,
    'Whether to visualize the GT fragment fields.')
flags.DEFINE_boolean(
    'vis_pred_frag_fields', False,
    'Whether to visualize the predicted fragment fields.')


def define_FLAGS():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string(
        'master', '',
        'BNS name of the tensorflow server')
    flags.DEFINE_boolean(
        'cpu_only', False,
        'Whether to run the inference on CPU only.')
    flags.DEFINE_string(
        'task_type', common.LOCALIZATION,  # LOCALIZATION, DETECTION
        'Type of the 6D object pose estimation task.')
    flags.DEFINE_list(
        'infer_tfrecord_names', None,
        'Names of tfrecord files (without suffix) used for inference.')
    flags.DEFINE_integer(
        'infer_max_height_before_crop', '1280',
        'Maximum image height before cropping (the image is downscaled if larger).')
    flags.DEFINE_list(
        'infer_crop_size', '720,1280',
        'Image size [height, width] during inference.')
    flags.DEFINE_string(
        'checkpoint_name', None,
        'Name of the checkpoint to evaluate (e.g. "model.ckpt-1000000"). The latest '
        'available checkpoint is used if None.')
    flags.DEFINE_boolean(
        'project_to_surface', False,
        'Whether to project the predicted 3D locations to the object model.')
    flags.DEFINE_boolean(
        'save_estimates', True,
        'Whether to save pose estimates in format expected by the BOP Challenge.')
    flags.DEFINE_boolean(
        'save_corresp', False,
        'Whether to save established correspondences to text files.')
    flags.DEFINE_string(
        'infer_name', None,
        'Name of the inference used in the filename of the saved estimates.')

    # Pose fitting parameters.
    flags.DEFINE_string(
        'fitting_method', common.PROGRESSIVE_X,  # PROGRESSIVE_X, OPENCV_RANSAC
        'Pose fitting method.')
    flags.DEFINE_float(
        'inlier_thresh', 4.0,
        'Tau_r in the CVPR 2020 paper. Inlier threshold [px] on the '
        'reprojection error.')
    flags.DEFINE_float(
        'neighbour_max_dist', 20.0,
        'Tau_d in the CVPR 2020 paper.')
    flags.DEFINE_float(
        'min_hypothesis_quality', 0.5,
        'Tau_q in the CVPR 2020 paper')
    flags.DEFINE_float(
        'required_progx_confidence', 0.5,
        'The required confidence used to calculate the number of Prog-X iterations.')
    flags.DEFINE_float(
        'required_ransac_confidence', 1.0,
        'The required confidence used to calculate the number of RANSAC iterations.')
    flags.DEFINE_float(
        'min_triangle_area', 0.0,
        'Tau_t in the CVPR 2020 paper.')
    flags.DEFINE_boolean(
        'use_prosac', False,
        'Whether to use the PROSAC sampler.')
    flags.DEFINE_integer(
        'max_model_number_for_pearl', 5,
        'Maximum number of instances to optimize by PEARL. PEARL is turned off if '
        'there are more instances to find.')
    flags.DEFINE_float(
        'spatial_coherence_weight', 0.1,
        'Weight of the spatial coherence in Graph-Cut RANSAC.')
    flags.DEFINE_float(
        'scaling_from_millimeters', 0.1,
        'Scaling factor of 3D coordinates when constructing the neighborhood graph. '
        '0.1 will convert mm to cm. See the CVPR 2020 paper for details.')
    flags.DEFINE_float(
        'max_tanimoto_similarity', 0.9,
        'See the Progressive-X paper.')
    flags.DEFINE_integer(
        'max_correspondences', None,
        'Maximum number of correspondences to use for fitting. Not applied if None.')
    flags.DEFINE_integer(
        'max_instances_to_fit', None,
        'Maximum number of instances to fit. Not applied if None.')
    flags.DEFINE_integer(
        'max_fitting_iterations', 400,
        'The maximum number of fitting iterations.')

    # Visualization parameters.
    flags.DEFINE_boolean(
        'vis', False,
        'Global switch for visualizations.')
    flags.DEFINE_boolean(
        'vis_gt_poses', False,
        'Whether to visualize the GT poses.')
    flags.DEFINE_boolean(
        'vis_pred_poses', True,
        'Whether to visualize the predicted poses.')
    flags.DEFINE_boolean(
        'vis_gt_obj_labels', False,
        'Whether to visualize the GT object labels.')
    flags.DEFINE_boolean(
        'vis_pred_obj_labels', True,
        'Whether to visualize the predicted object labels.')
    flags.DEFINE_boolean(
        'vis_pred_obj_confs', False,
        'Whether to visualize the predicted object confidences.')
    flags.DEFINE_boolean(
        'vis_gt_frag_fields', False,
        'Whether to visualize the GT fragment fields.')
    flags.DEFINE_boolean(
        'vis_pred_frag_fields', False,
        'Whether to visualize the predicted fragment fields.')

    return FLAGS
# ============================================================
# TensorFlow graph
# ============================================================

def build_epos_graph(num_objects):
    """
    Build the EPOS inference graph.

    The graph contains a dynamic RGB input. The actual image
    dimensions are obtained from the image at inference time.
    """

    # Load the EPOS parameters used by the trained model.
    common.update_flags(
        os.path.dirname(PARAMS_PATH)
    )

    with tf.Graph().as_default() as graph:

        # ----------------------------------------------------
        # Dynamic image input
        # ----------------------------------------------------
        #
        # [batch, height, width, channels]
        #
        # Height and width are intentionally not hardcoded.
        #
        image_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, None, 3],
            name="rgb_image_input"
        )

        # ----------------------------------------------------
        # Network output configuration
        # ----------------------------------------------------

        outputs_to_num_channels = (
            common.get_outputs_to_num_channels(
                num_objects,
                FLAGS.num_frags
            )
        )

        model_options = common.ModelOptions(
            outputs_to_num_channels=outputs_to_num_channels,
            crop_size=list(
                map(int, FLAGS.infer_crop_size)
            ),
            atrous_rates=FLAGS.atrous_rates,
            encoder_output_stride=FLAGS.encoder_output_stride
        )

        # ----------------------------------------------------
        # EPOS network
        # ----------------------------------------------------

        predictions = model.predict(
            images=tf.expand_dims(image_input, axis=0),
            model_options=model_options,
            upsample_logits=FLAGS.upsample_logits,
            image_pyramid=FLAGS.image_pyramid,
            num_objs=num_objects,
            num_frags=FLAGS.num_frags,
            frag_cls_agnostic=FLAGS.frag_cls_agnostic,
            frag_loc_agnostic=FLAGS.frag_loc_agnostic
        )

        return graph, image_input, predictions


# ============================================================
# Restore checkpoint and freeze graph
# ============================================================

def freeze_checkpoint(
    checkpoint_dir,
    frozen_pb_path,
    num_objects
):
    """
    Restore a TensorFlow checkpoint into the EPOS graph and
    freeze the variables into a single .pb file.
    """

    checkpoint_path = CHECKPOINT

    if checkpoint_path is None:
        checkpoint_path = tf.train.latest_checkpoint(
            checkpoint_dir
        )

    if checkpoint_path is None:
        raise RuntimeError(
            "No TensorFlow checkpoint found in: {}".format(
                checkpoint_dir
            )
        )

    print()
    print("Checkpoint:")
    print("  {}".format(checkpoint_path))

    graph, image_input, predictions = build_epos_graph(
        num_objects
    )

    # --------------------------------------------------------
    # Determine output tensors
    # --------------------------------------------------------
    #
    # These are the four outputs used by the original pipeline.
    #
    output_names = [
        "Softmax",
        "Reshape_1",
        "Softmax_1",
        "ArgMax",
    ]

    output_tensors = []

    for name in output_names:
        try:
            tensor = graph.get_tensor_by_name(
                name + ":0"
            )
        except KeyError:
            raise RuntimeError(
                "Could not find output tensor '{}:0'. "
                "Print graph operations to find the correct "
                "EPOS output names.".format(name)
            )

        output_tensors.append(tensor)

    # --------------------------------------------------------
    # Restore checkpoint
    # --------------------------------------------------------

    saver = tf.train.Saver(
        var_list=misc.get_variable_dict()
    )

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.intra_op_parallelism_threads = 10
    config_proto.inter_op_parallelism_threads = 10

    os.makedirs(
        os.path.dirname(frozen_pb_path),
        exist_ok=True
    )

    with tf.Session(
        graph=graph,
        config=config_proto
    ) as sess:

        print("Restoring checkpoint...")

        saver.restore(
            sess,
            checkpoint_path
        )

        print("Checkpoint restored.")

        # ----------------------------------------------------
        # Freeze variables
        # ----------------------------------------------------

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_names
        )

        frozen_graph = tf.graph_util.remove_training_nodes(
            frozen_graph
        )

        with tf.gfile.GFile(
            frozen_pb_path,
            "wb"
        ) as f:
            f.write(
                frozen_graph.SerializeToString()
            )

        print()
        print("Frozen graph written to:")
        print("  {}".format(frozen_pb_path))

    return frozen_pb_path


# ============================================================
# TensorFlow PB -> ONNX
# ============================================================

def convert_pb_to_onnx(
    frozen_pb_path,
    onnx_path
):
    """
    Convert the frozen TensorFlow graph to ONNX.
    """

    print()
    print("Converting PB -> ONNX...")

    with tf.gfile.GFile(
        frozen_pb_path,
        "rb"
    ) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(
            f.read()
        )

    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=[
            "rgb_image_input:0"
        ],
        output_names=[
            "Softmax:0",
            "Reshape_1:0",
            "Softmax_1:0",
            "ArgMax:0"
        ],
        opset=11,
        output_path=onnx_path
    )

    print("ONNX model written to:")
    print("  {}".format(onnx_path))

    return onnx_path


# ============================================================
# ONNX inference
# ============================================================

def load_onnx_model(onnx_path):
    """
    Load the ONNX model using CUDA if available.
    """

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_use_max_workspace": "1",
                "cudnn_conv_algo_search": "DEFAULT",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
            }
        ),
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(
        onnx_path,
        providers=providers
    )

    print()
    print("ONNX input:")
    print(session.get_inputs()[0])

    print()
    print("ONNX outputs:")

    for output in session.get_outputs():
        print(
            "  {} {}".format(
                output.name,
                output.shape
            )
        )

    return session


def run_network(
    session,
    image
):
    """
    Run the EPOS network on one image.

    image:
        H x W x 3 float32 RGB image.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [H, W, 3], got {}".format(
                image.shape
            )
        )

    if image.shape[-1] != 3:
        raise ValueError(
            "Expected RGB image, got {}".format(
                image.shape
            )
        )

    input_name = session.get_inputs()[0].name

    # EPOS model expects a batch dimension.
    batch = np.expand_dims(
        image,
        axis=0
    )

    # Warmup.
    session.run(
        None,
        {
            input_name: batch
        }
    )

    start = time.time()

    outputs = session.run(
        None,
        {
            input_name: batch
        }
    )

    elapsed = time.time() - start

    print()
    print("Network inference: {:.4f} s".format(elapsed))

    return outputs, elapsed


# ============================================================
# Camera
# ============================================================

def make_camera_matrix():
    return np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32
    )


# ============================================================
# EPOS model store
# ============================================================

def create_model_store():
    """
    Load the EPOS object models and fragmentation data.
    """

    model_dir = os.path.join(
        config.TF_MODELS_PATH,
        FLAGS.model
    )

    frag_path = os.path.join(
        model_dir,
        "fragments.pkl"
    )

    if not os.path.exists(frag_path):

        print(
            "Fragmentation file not found. "
            "Creating fragmentation..."
        )

        model_store = datagen.ObjectModelStore(
            dataset_name="essential2",
            model_type="cad",
            num_frags=FLAGS.num_frags,
            prepare_for_projection=FLAGS.project_to_surface
        )

        model_store.fragment_models()

        frag_centers = model_store.frag_centers
        frag_sizes = model_store.frag_sizes

    else:

        print(
            "Loading fragmentation from:"
        )
        print(
            "  {}".format(frag_path)
        )

        with open(
            frag_path,
            "rb"
        ) as f:
            fragments = pickle.load(f)

        frag_centers = fragments["frag_centers"]
        frag_sizes = fragments["frag_sizes"]

    model_store = datagen.ObjectModelStore(
        dataset_name="essential2",
        model_type="cad",
        num_frags=FLAGS.num_frags,
        frag_centers=frag_centers,
        frag_sizes=frag_sizes,
        prepare_for_projection=FLAGS.project_to_surface
    )

    model_store.load_models()

    return model_store


# ============================================================
# Correspondences + PnP
# ============================================================

def estimate_pose(
    outputs,
    image,
    K,
    model_store
):
    """
    Convert EPOS predictions into 6D poses.

    This is the simplified version of your original
    process_image() function.
    """

    pred_frag_conf = outputs[0]
    pred_frag_loc = outputs[1]
    pred_obj_conf = outputs[2]
    pred_obj_label = outputs[3]

    # --------------------------------------------------------
    # Build correspondence dictionary
    # --------------------------------------------------------

    correspondences = corresp.establish_many_to_many(
        obj_confs=pred_obj_conf[0],
        frag_confs=pred_frag_conf[0],
        frag_coords=pred_frag_loc[0],
        model_store=model_store,
        output_scale=(
            1.0 / FLAGS.decoder_output_stride[0]
            if not FLAGS.upsample_logits
            else 1.0
        ),
        min_obj_conf=FLAGS.corr_min_obj_conf,
        min_frag_rel_conf=FLAGS.corr_min_frag_rel_conf,
        project_to_surface=FLAGS.project_to_surface,
        only_annotated_objs=(
            FLAGS.task_type == common.LOCALIZATION
        )
    )

    poses = []

    # --------------------------------------------------------
    # Fit one pose per object.
    # --------------------------------------------------------

    for obj_id, obj_corr in correspondences.items():

        coord_2d = np.asarray(
            obj_corr["coord_2d"],
            dtype=np.float64
        )

        coord_3d = np.asarray(
            obj_corr["coord_3d"],
            dtype=np.float64
        )

        if len(coord_2d) < 6:
            print(
                "Skipping object {}: only {} correspondences.".format(
                    obj_id,
                    len(coord_2d)
                )
            )
            continue

        # Highest-confidence correspondences first.
        order = np.argsort(
            obj_corr["conf"]
        )[::-1]

        coord_2d = np.ascontiguousarray(
            coord_2d[order]
        )

        coord_3d = np.ascontiguousarray(
            coord_3d[order]
        )

        if FLAGS.max_correspondences is not None:

            coord_2d = coord_2d[
                :FLAGS.max_correspondences
            ]

            coord_3d = coord_3d[
                :FLAGS.max_correspondences
            ]

        # ----------------------------------------------------
        # OpenCV PnP-RANSAC
        # ----------------------------------------------------

        success, rvec, tvec, inliers = (
            cv2.solvePnPRansac(
                objectPoints=coord_3d,
                imagePoints=coord_2d,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=FLAGS.max_fitting_iterations,
                reprojectionError=FLAGS.inlier_thresh,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP
            )
        )

        if not success:
            print(
                "Pose estimation failed for object {}".format(
                    obj_id
                )
            )
            continue

        R, _ = cv2.Rodrigues(rvec)

        poses.append(
            {
                "obj_id": obj_id,
                "R": R,
                "t": tvec,
                "score": 0.0,
                "num_inliers": (
                    len(inliers)
                    if inliers is not None
                    else 0
                )
            }
        )

    return poses


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Freeze checkpoint
    # --------------------------------------------------------

    freeze_checkpoint(
        checkpoint_dir=MODEL_DIR,
        frozen_pb_path=PB_PATH,
        num_objects=NUM_OBJECTS
    )

    # --------------------------------------------------------
    # 2. Convert frozen PB -> ONNX
    # --------------------------------------------------------

    convert_pb_to_onnx(
        frozen_pb_path=PB_PATH,
        onnx_path=ONNX_PATH
    )

    # --------------------------------------------------------
    # 3. Load ONNX
    # --------------------------------------------------------

    session = load_onnx_model(
        ONNX_PATH
    )

    # --------------------------------------------------------
    # 4. Load image
    # --------------------------------------------------------

    image = np.asarray(
        Image.open(IMAGE_PATH).convert("RGB"),
        dtype=np.float32
    )

    print()
    print("Image shape:")
    print("  {}".format(image.shape))

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # 5. Camera matrix
    # --------------------------------------------------------

    K = make_camera_matrix()

    print()
    print("Camera matrix:")
    print(K)

    # --------------------------------------------------------
    # 6. Network inference
    # --------------------------------------------------------

    outputs, inference_time = run_network(
        session,
        image
    )

    # --------------------------------------------------------
    # 7. Load EPOS object models
    # --------------------------------------------------------

    model_store = create_model_store()

    # --------------------------------------------------------
    # 8. Estimate pose
    # --------------------------------------------------------

    start = time.time()

    poses = estimate_pose(
        outputs=outputs,
        image=image,
        K=K,
        model_store=model_store
    )

    pose_time = time.time() - start

    # --------------------------------------------------------
    # 9. Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        "Image: {} x {}".format(
            width,
            height
        )
    )

    print(
        "Inference: {:.4f} s".format(
            inference_time
        )
    )

    print(
        "Pose estimation: {:.4f} s".format(
            pose_time
        )
    )

    print(
        "Objects detected: {}".format(
            len(poses)
        )
    )

    for i, pose in enumerate(poses):

        print()
        print("Pose {}".format(i))
        print("Object ID:", pose["obj_id"])
        print("Inliers:", pose["num_inliers"])

        print("R:")
        print(pose["R"])

        print("t:")
        print(pose["t"])


if __name__ == "__main__":

    # EPOS uses tf.app.flags internally.
    #
    # Your existing params file is still responsible for
    # things such as:
    #
    #   num_frags
    #   upsample_logits
    #   decoder_output_stride
    #   encoder_output_stride
    #   atrous_rates
    #   frag_cls_agnostic
    #   frag_loc_agnostic
    #   etc.
    #
    # Make sure those correspond to the checkpoint.

    main()
