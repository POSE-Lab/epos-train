import tensorflow as tf
from PIL import Image
from epos_lib import model
import numpy as np
from epos_lib import common  # utility functions
from epos_lib import misc
import os
import pickle
from epos_lib import datagen
import tf2onnx
import onnxruntime as nxrun
import time
from epos_lib import corresp
from bop_toolkit_lib import inout
from epos_lib import vis
from bop_toolkit_lib import visualization
import pyprogressivex
import cv2
import bop_renderer
from bop_toolkit_lib import dataset_params 
from epos_lib import config
import onnxruntime as rt

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

def visualize(
        inputs, predictions, pred_poses, im_ind, crop_size, output_scale,
        model_store, renderer, vis_dir):
    """Visualizes estimates from one image.

    Args:
      samples: Dictionary with input data.
      predictions: Dictionary with predictions.
      pred_poses: Predicted poses.
      im_ind: Image index.
      crop_size: Image crop size (width, height).
      output_scale: Scale of the model output w.r.t. the input (output / input).
      model_store: Store for 3D object models of class ObjectModelStore.
      renderer: Renderer of class bop_renderer.Renderer().
      vis_dir: Directory where the visualizations will be saved.
    """
    tf.logging.info('Visualization for: {}'.format(
        inputs["image_path"]))

    # Size of a visualization grid tile.
    tile_size = (300, 225)

    # Extension of the saved visualizations ('jpg', 'png', etc.).
    vis_ext = 'jpg'

    # Font settings.
    font_size = 10
    font_color = (0.8, 0.8, 0.8)

    # Intrinsics.
    K = inputs["K"]
    output_K = K * output_scale
    output_K[2, 2] = 1.0

    # Tiles for the grid visualization.
    tiles = []

    # Size of the output fields.
    output_size =\
        int(output_scale * crop_size[0]), int(output_scale * crop_size[1])

    # Prefix of the visualization names.
    vis_prefix = '{:06d}'.format(im_ind)

    # Input RGB image.
    rgb = np.squeeze(inputs["image"])
    vis_rgb = visualization.write_text_on_image(
        misc.resize_image_py(rgb, tile_size).astype(np.uint8),
        [{'name': '', 'val': 'input', 'fmt': ':s'}],
        size=font_size, color=font_color)
    tiles.append(vis_rgb)

    # Visualize the estimated poses.
    if FLAGS.vis_pred_poses:
        vis_pred_poses = vis.visualize_object_poses(
            rgb, K, pred_poses, renderer)
        vis_pred_poses = visualization.write_text_on_image(
            misc.resize_image_py(vis_pred_poses, tile_size),
            [{'name': '', 'val': 'pred poses', 'fmt': ':s'}],
            size=font_size, color=font_color)
        tiles.append(vis_pred_poses)

    # Predicted object labels.
    if FLAGS.vis_pred_obj_labels:
        obj_labels = np.squeeze(predictions[common.PRED_OBJ_LABEL][0])
        obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
        obj_labels = vis.colorize_label_map(obj_labels)
        obj_labels = visualization.write_text_on_image(
            misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
            [{'name': '', 'val': 'predicted obj labels', 'fmt': ':s'}],
            size=font_size, color=font_color)
        tiles.append(obj_labels)

    # Predicted object confidences.
    if FLAGS.vis_pred_obj_confs:
        num_obj_labels = predictions[common.PRED_OBJ_CONF].shape[-1]
        for obj_label in range(num_obj_labels):
            obj_confs = misc.resize_image_py(np.array(
                predictions[common.PRED_OBJ_CONF][0, :, :, obj_label]), tile_size)
            obj_confs = (255.0 * obj_confs).astype(np.uint8)
            obj_confs = np.dstack([obj_confs, obj_confs, obj_confs])  # To RGB.
            obj_confs = visualization.write_text_on_image(
                obj_confs, [{'name': 'cls', 'val': obj_label, 'fmt': ':d'}],
                size=font_size, color=font_color)
            tiles.append(obj_confs)

    # Visualization of predicted fragment fields.
    if FLAGS.vis_pred_frag_fields:
        vis.visualize_pred_frag(
            frag_confs=predictions[common.PRED_FRAG_CONF][0],
            frag_coords=predictions[common.PRED_FRAG_LOC][0],
            output_size=output_size,
            model_store=model_store,
            vis_prefix=vis_prefix,
            vis_dir=vis_dir,
            vis_ext=vis_ext)

    # Build and save a visualization grid.
    grid = vis.build_grid(tiles, tile_size)
    grid_vis_path = os.path.join(
        vis_dir, '{}_grid.{}'.format(vis_prefix, vis_ext))
    inout.save_im(grid_vis_path, grid)


def process_image(
        inputs, predictions,results, im_ind, crop_size, output_scale, model_store,
        renderer, task_type, infer_name, infer_dir, vis_dir):
    """Estimates object poses from one image.

    Args:
      sess: TensorFlow session.
      samples: Dictionary with input data.
      predictions: Dictionary with predictions.
      im_ind: Index of the current image.
      crop_size: Image crop size (width, height).
      output_scale: Scale of the model output w.r.t. the input (output / input).
      model_store: Store for 3D object models of class ObjectModelStore.
      renderer: Renderer of class bop_renderer.Renderer().
      task_type: 6D object pose estimation task (common.LOCALIZATION or
        common.DETECTION).
      infer_name: Name of the current inference.
      infer_dir: Folder for inference results.
      vis_dir: Folder for visualizations.
    """
    # Dictionary for run times.
    run_times = {}
    keys = [common.PRED_FRAG_CONF,common.PRED_FRAG_LOC,common.PRED_OBJ_CONF,common.PRED_OBJ_LABEL]
    predictions = dict(zip(keys,results))
    # Prediction.
    time_start = time.time()
    run_times['prediction'] = time.time() - time_start
    print(predictions)
    # Scene and image ID's.
    scene_id = inputs["scene_id"]
    im_id = inputs["im_id"]

    # Intrinsic parameters.
    K = inputs["K"]

    gt_poses = None

    # Establish many-to-many 2D-3D correspondences.
    time_start = time.time()
    corr = corresp.establish_many_to_many(
        obj_confs=predictions[common.PRED_OBJ_CONF][0],
        frag_confs=predictions[common.PRED_FRAG_CONF][0],
        frag_coords=predictions[common.PRED_FRAG_LOC][0],
        model_store=model_store,
        output_scale=output_scale,
        min_obj_conf=FLAGS.corr_min_obj_conf,
        min_frag_rel_conf=FLAGS.corr_min_frag_rel_conf,
        project_to_surface=FLAGS.project_to_surface,
        only_annotated_objs=(task_type == common.LOCALIZATION))
    run_times['establish_corr'] = time.time() - time_start

    # PnP-RANSAC to estimate 6D object poses from the correspondences.
    time_start = time.time()
    poses = []
    for obj_id, obj_corr in corr.items():
        # tf.logging.info(
        #   'Image path: {}, obj: {}'.format(samples[common.IMAGE_PATH][0], obj_id))

        # Number of established correspondences.
        num_corrs = obj_corr['coord_2d'].shape[0]

        # Skip the fitting if there are too few correspondences.
        min_required_corrs = 6
        if num_corrs < min_required_corrs:
            continue

        # The correspondences need to be sorted for PROSAC.
        if FLAGS.use_prosac:
            sorted_inds = np.argsort(obj_corr['conf'])[::-1]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][sorted_inds]

        # Select correspondences with the highest confidence.
        if FLAGS.max_correspondences is not None \
                and num_corrs > FLAGS.max_correspondences:
            # Sort the correspondences only if they have not been sorted for PROSAC.
            if FLAGS.use_prosac:
                keep_inds = np.arange(num_corrs)
            else:
                keep_inds = np.argsort(obj_corr['conf'])[::-1]
            keep_inds = keep_inds[:FLAGS.max_correspondences]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][keep_inds]

        # Make sure the coordinates are saved continuously in memory.
        coord_2d = np.ascontiguousarray(
            obj_corr['coord_2d'].astype(np.float64))
        coord_3d = np.ascontiguousarray(
            obj_corr['coord_3d'].astype(np.float64))

        if FLAGS.fitting_method == common.PROGRESSIVE_X:
            # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
            # Progressive-X is applied and up to num_instances poses are returned.
            # If num_instances == -1, then Progressive-X is applied and all found
            # poses are returned.
            if task_type == common.LOCALIZATION:
                num_instances = 1
            else:
                num_instances = -1

            if FLAGS.max_instances_to_fit is not None:
                num_instances = min(num_instances, FLAGS.max_instances_to_fit)

            pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
                x1y1=coord_2d,
                x2y2z2=coord_3d,
                K=K,
                threshold=FLAGS.inlier_thresh,
                neighborhood_ball_radius=FLAGS.neighbour_max_dist,
                spatial_coherence_weight=FLAGS.spatial_coherence_weight,
                scaling_from_millimeters=FLAGS.scaling_from_millimeters,
                max_tanimoto_similarity=FLAGS.max_tanimoto_similarity,
                max_iters=FLAGS.max_fitting_iterations,
                conf=FLAGS.required_progx_confidence,
                proposal_engine_conf=FLAGS.required_ransac_confidence,
                min_coverage=FLAGS.min_hypothesis_quality,
                min_triangle_area=FLAGS.min_triangle_area,
                min_point_number=6,
                max_model_number=num_instances,
                max_model_number_for_optimization=FLAGS.max_model_number_for_pearl,
                use_prosac=FLAGS.use_prosac,
                log=False)

            pose_est_success = pose_ests is not None
            if pose_est_success:
                for i in range(int(pose_ests.shape[0] / 3)):
                    j = i * 3
                    R_est = pose_ests[j:(j + 3), :3]
                    t_est = pose_ests[j:(j + 3), 3].reshape((3, 1))
                    poses.append({
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'R': R_est,
                        't': t_est,
                        'score': pose_qualities[i],
                    })

        elif FLAGS.fitting_method == common.OPENCV_RANSAC:
            # This integration of OpenCV-RANSAC can estimate pose of only one object
            # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
            # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
            # in the Progressive-X scheme (as the other methods in that table).
            pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
                objectPoints=coord_3d,
                imagePoints=coord_2d,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=FLAGS.max_fitting_iterations,
                reprojectionError=FLAGS.inlier_thresh,
                confidence=0.99,  # FLAGS.required_ransac_confidence
                flags=cv2.SOLVEPNP_EPNP)

            if pose_est_success:
                poses.append({
                    'scene_id': scene_id,
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'R': cv2.Rodrigues(r_est)[0],
                    't': t_est,
                    'score': 0.0,  # TODO: Define the score.
                })

        else:
            raise ValueError(
                'Unknown pose fitting method ({}).'.format(FLAGS.fitting_method))

    run_times['fitting'] = time.time() - time_start
    run_times['total'] = np.sum(list(run_times.values()))

    # Add the total time to each pose.
    for pose in poses:
        pose['time'] = run_times['total']

    # Visualization.
    if FLAGS.vis:
        visualize(
            inputs=inputs,
            predictions=predictions,
            pred_poses=poses,
            im_ind=im_ind,
            crop_size=crop_size,
            output_scale=output_scale,
            model_store=model_store,
            renderer=renderer,
            vis_dir=vis_dir)

    return poses, run_times
def input_sample():  # partial copy of datagen._parse_and_preprocess()

    # input image converted into tensor
   
    im = tf.placeholder(
        shape=(720, 1280,3), dtype=tf.float32, name="rgb_image_input")

    



    # K matrix
    # !!!FOR THE CROPED IMAGE!!!
    fx = 640.9191284179688  # * im_scale
    fy = 639.4614868164062  # * im_scale
    cx = 631.1490478515625  # * im_scale
    cy = 363.1187744140625  # * im_scale

    # K = tf.convert_to_tensor(
    #     [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], name='K_input')

    # define the output dictionary sample
    # scene id set to 1 since we don't care about it we only infer one image
    # same thing for the IM_ID

    sample = {
        # "scene_id": tf.convert_to_tensor(1, dtype=tf.int64, name="scene_id_input"),
        # "im_id": tf.convert_to_tensor(1, dtype=tf.int64, name='im_id_input'),
        # "image_path": tf.convert_to_tensor(image_path, dtype=tf.string, name='image_path_input'),
        "image": im
        # "K": K,
    }
    # note that the sample dictionary cotains more tensors regarding the gt poses and annotations
    # since we care about the inference they have been removed
    return sample


def create_infer_graph(model_path, image_size, objects):

    common.update_flags(os.path.join(
        "/home/store/tf_models/essential2/", common.PARAMS_FILENAME))

    if FLAGS.upsample_logits:
        # The stride is 1 if the logits are upsampled to the input resolution.
        output_stride = 1
    else:
        assert (len(FLAGS.decoder_output_stride) == 1)
        output_stride = FLAGS.decoder_output_stride[0]

    with tf.Graph().as_default() as graph:

        # img_tensor = tf.constant(

        outputs_to_num_channels = common.get_outputs_to_num_channels(
            objects, FLAGS.num_frags)

        # options hard-coded from yaml parameteres file in model/train
        model_options = common.ModelOptions(
            outputs_to_num_channels=outputs_to_num_channels,
            crop_size=list(map(int, FLAGS.infer_crop_size)),
            atrous_rates=FLAGS.atrous_rates,
            encoder_output_stride=FLAGS.encoder_output_stride)

        sample = input_sample()
        predictions = model.predict(
            images=tf.expand_dims(sample[common.IMAGE], axis=0),
            model_options=model_options,
            upsample_logits=FLAGS.upsample_logits,
            image_pyramid=FLAGS.image_pyramid,
            num_objs=objects,
            num_frags=FLAGS.num_frags,
            frag_cls_agnostic=FLAGS.frag_cls_agnostic,
            frag_loc_agnostic=FLAGS.frag_loc_agnostic)

        checkpoint_path = tf.train.latest_checkpoint(model_path)
        tf.logging.info('Inference with model: {}'.format(checkpoint_path))
        tf_config = tf.ConfigProto()

        # tf_config.gpu_options.allow_growth = True  # Only necessary GPU memory.
        tf_config.gpu_options.allow_growth = False
        tf_config.intra_op_parallelism_threads = 10

        # Scaffold for initialization.
        scaffold = tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            saver=tf.train.Saver(var_list=misc.get_variable_dict()))

        tf_config.inter_op_parallelism_threads = 10
        session_creator = tf.train.ChiefSessionCreator(
            config=tf_config,
            scaffold=scaffold,
            master=FLAGS.master,
            checkpoint_filename_with_path=checkpoint_path)

        with tf.train.MonitoredSession(
                session_creator=session_creator, hooks=None) as sess:

            output_node_names = "Softmax,Reshape_1,Softmax_1,ArgMax"
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, tf.get_default_graph().as_graph_def(), output_node_names.split(','))
            output_graph_def = tf.graph_util.remove_training_nodes(
                output_graph_def, protected_nodes=None)
            with tf.gfile.GFile("/home/store/tf_models/essential2/frozen/essential2_frozen_1280_1450000.pb", 'wb') as f:
                f.write(output_graph_def.SerializeToString())

            # save inference graph
            tf.train.write_graph(
                graph, "/home/store/tf_models/essential2/frozen", "essential2_frozen_1280_1450000.pbtxt", as_text=True)
    return predictions

def freeze_graph():
    pass


def load_graph(fielpath):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with tf.gfile.FastGFile(fielpath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    image_t = [tensor for op in graph.get_operations()
               for tensor in op.values()]
    print("TENSORSSSSSS")
    # define the input tensors
    # input_im_id = tf.placeholder(
    #     dtype=tf.int64, shape=[], name="im_id_input")
    # input_scene_id = tf.placeholder(
    #     dtype=tf.int64, shape=[], name="scene_id_input")
    # input_image_path = tf.placeholder(
    #     dtype=tf.string, shape=[], name="image_path_input")
    # input_K = tf.placeholder(dtype=tf.float32, shape=[
    #     3, 3], name="K_input")

    tf.import_graph_def(graph_def, name='')
    # tf.train.write_graph(
    #             graph, "/home/panos/STORE/tf_models/extendComp_object13/test_graphs", "final_removed_un.pb", as_text=True)
    #convert to onnx
    #model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,input_names=["im_id_input:0","scene_id_input:0","image_path_input:0","rgb_image_input:0","K_input:0"],
    #output_names=["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],output_path="/home/panos/STORE/tf_models/extendComp_object13/test_graphs")
    return sess, graph


def infer(sess, graph):

    
    fx = 640.9191284179688  # * im_scale
    fy = 639.4614868164062  # * im_scale
    cx = 631.1490478515625  # * im_scale
    cy = 363.1187744140625  # * im_scale
    all_ops = tf.get_default_graph().get_operations()
    print("TENSORSSSSSS")
    # print("TENSOR_IMAGE = ",type(input_im_id))
    print(all_ops)
    # tf.train.write_graph(
    #             graph, "/home/panos/STORE/tf_models/crfAndLab12/frozen_graphs/", "final_TESTT.pb", as_text=False)
    model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph.as_graph_def(),input_names=["rgb_image_input:0"],opset=11,
    output_names=["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],output_path="/home/store/tf_models/essential2/frozen/essential2_1280_1450000.onnx")

    # uncomment to infer

    # output_tensors = {'pred_frag_conf': graph.get_tensor_by_name("Softmax:0"),
    #                   'pred_frag_loc': graph.get_tensor_by_name("Reshape_1:0"),
    #                   'pred_obj_conf': graph.get_tensor_by_name("Softmax_1:0"),
    #                   'pred_obj_label': graph.get_tensor_by_name("ArgMax:0")}
    # warmup_start = time.time()
    # output = sess.run(output_tensors, feed_dict={
    #                                              graph.get_tensor_by_name("rgb_image_input:0"): Image.open('/home/panos/SPREADER_DATASET/felice/train_primesense/000001/rgb/000003.png')})
    # warmup_end = time.time() - warmup_start
    # time_infer_start = time.time()
    # output = sess.run(output_tensors, feed_dict={
    #                                              graph.get_tensor_by_name("rgb_image_input:0"): Image.open('/home/panos/SPREADER_DATASET/felice/train_primesense/000001/rgb/000003.png')})
    # time_infer_end = time.time() - time_infer_start
    # print(output)
    # print("Time infering took : ",time_infer_end, "Warmup took: ",warmup_end)
    
def onnx_infer(onnx_filename,predictions):

    model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)
    if FLAGS.upsample_logits:
        # The stride is 1 if the logits are upsampled to the input resolution.
        output_stride = 1
    else:
        assert (len(FLAGS.decoder_output_stride) == 1)
        output_stride = FLAGS.decoder_output_stride[0]
 # visualization initialization
        # initialize renderer #TODO: make it optional
    renderer = None
    if FLAGS.vis:
        tf.logging.info('Initializing renderer for visualization...')
        renderer = bop_renderer.Renderer()
        renderer.init(1280, 720)

            # load model info

        model_type_vis = None
        dp_model = dataset_params.get_model_params(
                config.BOP_PATH, 'essential2', model_type=model_type_vis)

        for obj_id in dp_model['obj_ids']:
            path = dp_model['model_tpath'].format(obj_id=obj_id)
            renderer.add_object(obj_id, path)

        tf.logging.info('Renderer initialized.')

    frag_path = os.path.join(model_dir, 'fragments.pkl')
    if os.path.exists(frag_path):
        tf.logging.info('Loading fragmentation from: {}'.format(frag_path))

        with open(frag_path, 'rb') as f:
            fragments = pickle.load(f)
            frag_centers = fragments['frag_centers']
            frag_sizes = fragments['frag_sizes']

        # Check if the loaded fragmentation is valid.
        for obj_id in frag_centers.keys():
            if frag_centers[obj_id].shape[0] != FLAGS.num_frags\
                    or frag_sizes[obj_id].shape[0] != FLAGS.num_frags:
                raise ValueError('The loaded fragmentation is not valid.')

    else:
        tf.logging.info(
            'Fragmentation does not exist (expected file: {}).'.format(frag_path))
        tf.logging.info('Calculating fragmentation...')

        model_type_frag_str = 'cad'
        if model_type_frag_str is None:
            model_type_frag_str = 'original'
        tf.logging.info('Type of models: {}'.format(model_type_frag_str))

    # Load 3D object models for fragmentation.
        model_store_frag = datagen.ObjectModelStore(
            dataset_name='essential2',
            model_type='cad',
            num_frags=FLAGS.num_frags,
            prepare_for_projection=FLAGS.project_to_surface)

        # Fragment the 3D object models.
        model_store_frag.fragment_models()
        frag_centers = model_store_frag.frag_centers
        frag_sizes = model_store_frag.frag_sizes

    # Load 3D object models for rendering.
    model_store = datagen.ObjectModelStore(
            dataset_name='essential2',
            model_type='cad',
            num_frags=FLAGS.num_frags,
            frag_centers=frag_centers,
            frag_sizes=frag_sizes,
            prepare_for_projection=FLAGS.project_to_surface)
    model_store.load_models()

    outputs_to_num_channels = common.get_outputs_to_num_channels(
            2, FLAGS.num_frags)

    # options hard-coded from yaml parameteres file in model/train
    model_options = common.ModelOptions(
            outputs_to_num_channels=outputs_to_num_channels,
            crop_size=list(map(int, FLAGS.infer_crop_size)),
            atrous_rates=FLAGS.atrous_rates,
            encoder_output_stride=FLAGS.encoder_output_stride)

    providers= [("CUDAExecutionProvider", 
        {"cudnn_conv_use_max_workspace": "1",
         "cudnn_conv_algo_search": "DEFAULT",
         "gpu_mem_limit": 2 * 1024 * 1024 * 1024}),"CPUExecutionProvider"]

    time_l_start = time.time()
    sess = nxrun.InferenceSession(onnx_filename,providers=providers)
    time_l_f = time.time() - time_l_start

    print("Loading of model took :",time_l_f)

    print("The model expects input shape: ", sess.get_inputs()[0])

    fx = 640.9191284179688  # * im_scale
    fy = 639.4614868164062  # * im_scale
    cx = 631.1490478515625  # * im_scale
    cy = 363.1187744140625  # * im_scale

    image_id = 1
    scene_id = 1
    image_path = "lala"
    image = Image.open('/home/datasets/essential2/val/000000/rgb_realsense/000001.png')

    warmup_Start = time.time()
    result = sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
    {"rgb_image_input:0":np.array(image).astype(np.float32)})
    warmup_end = time.time() - warmup_Start
    print("Time infering took : ",warmup_end)

    time_infer_start = time.time()
    result = sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
    {"rgb_image_input:0":np.array(image).astype(np.float32)})
    time_infer_end = time.time() - time_infer_start
    print("Time infering took : ",time_infer_end)

    inputs={"im_id":np.array((1)).astype(np.int64),"scene_id":np.array((1)).astype(np.int64),"image_path":np.array('/home/datasets/essential2/val/000000/rgb_realsense/000001.png'),
    "image":np.array(image).astype(np.float32),"K":np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).astype(np.float32)}
    poses,run_times = process_image(inputs=inputs,
                predictions=predictions,
                results=result,
                im_ind=0,
                crop_size=list(map(int, FLAGS.infer_crop_size)),
                output_scale=(1.0 / output_stride),
                model_store=model_store,
                renderer=renderer,
                task_type=FLAGS.task_type,
                infer_name=FLAGS.infer_name,
                infer_dir="/home/store/tf_models/essential_test/test_graphs",
                vis_dir="/home/store/tf_models/essential_test/test_graphs")
    print('Image: {}, prediction: {:.3f}, establish_corr: {:.3f}, '
                                    'fitting: {:.3f}, total time: {:.3f}'.format(
                                        0, time_infer_end, run_times['establish_corr'],
                                        run_times['fitting'], run_times['total']))
    print(result)
    
if __name__ == "__main__":

    print("versionnnnnnnnnnnnnnnnnnnnnnn")
    print(rt.cuda_version)
    # FLAGS = define_FLAGS()
    predictions = create_infer_graph(
        "/home/store/tf_models/essential2/train", (720, 1280, 3), 1)  # working

    # # # working

    sess, graph = load_graph(
            "/home/store/tf_models/essential2/frozen/essential2_frozen_1280_1450000.pb")

    infer(sess, graph)

    onnx_infer("/home/store/tf_models/essential2/frozen/essential2_1280_1450000.onnx",predictions)
