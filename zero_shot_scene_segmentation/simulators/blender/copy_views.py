import os
import shutil
import sys
import argparse



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='sample_views',
                    usage='python <path to copy_views.py> -- [options]',
                    description='Python script for copying view samples between machines',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-src', '--source-dataset-dir', help='path to directory where existing dataset with sampled views exists', type=str)
    parser.add_argument('-dst', '--destination-dataset-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)

    if args.verbose:
        print()
        print(args)
        print()

    scene_directories = [path for path in os.listdir(args.source_dataset_dir) if os.path.isdir(os.path.join(args.source_dataset_dir, path))]
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.source_dataset_dir, scene_dir)

        scene_files = os.listdir(scene_dir_path)

        scene_out_path = os.path.join(args.destination_dataset_dir, scene_dir)
        scene_view_all_poses_path = scene_dir+'.all_view_poses.csv'
        scene_view_accepted_poses_path = scene_dir+'.accepted_view_poses.csv'

        scene_has_sampled_views = os.path.isfile(os.path.join(scene_dir_path, scene_view_all_poses_path)) and os.path.isfile(os.path.join(scene_dir_path, scene_view_accepted_poses_path))

        if scene_has_sampled_views:
            os.makedirs(scene_out_path, exist_ok=True)
            shutil.copyfile(os.path.join(scene_dir_path, scene_view_all_poses_path), os.path.join(scene_out_path, scene_view_all_poses_path))
            shutil.copyfile(os.path.join(scene_dir_path, scene_view_accepted_poses_path), os.path.join(scene_out_path, scene_view_accepted_poses_path))

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE COPYING ALL SCENES")
        print("***********************")
        print()